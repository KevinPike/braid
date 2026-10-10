"""M15 checkpointing at the graph level: SQLite saver, resume after a cancel, the Recorder inside the checkpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from harness.config import WorkflowConfig
from harness.sandbox import LocalSandbox, Workspace
from harness.workflow.execute import build_execution_graph, drive_graph, initial_state
from harness.workflow.plan import Plan
from harness.workflow.records import BehaviorStats, Recorder
from harness.workflow.runs import RunInfo, RunStore, open_checkpointer
from tests.workflow.test_execute import MUL_IMPL, MUL_TEST, ScriptedModel, edit, MUL, tree


class Hangs:
    """A scripted model whose call number ``at`` (0-based) never returns, like a model call cut off by Ctrl-C."""

    def __init__(self, replies: list[Any], at: int) -> None:
        self.inner = ScriptedModel(replies)
        self.at = at
        self.calls = 0

    async def complete(self, system: str, prompt: str, schema: Any) -> Any:
        self.calls += 1
        if self.calls - 1 == self.at:
            await asyncio.Event().wait()
        return await self.inner.complete(system, prompt, schema)


def test_the_recorder_round_trips_through_dump_and_load() -> None:
    rec = Recorder(behaviors=[BehaviorStats("a", green=True, attempts=2, seconds=1.5), BehaviorStats("b", satisfied=True)])
    rec.trace += ["write_test", "run_red"]
    rec.stuck, rec.polish, rec.model_calls, rec.model_seconds = 1, "kept", 4, 2.5
    data = rec.dump()
    other = Recorder()
    other.load(data)
    assert other.behaviors == rec.behaviors and other.trace == rec.trace
    assert (other.stuck, other.polish, other.model_calls, other.model_seconds) == (1, "kept", 4, 2.5)
    assert abs(other.elapsed_s - rec.elapsed_s) < 0.5  # the clock carries on from where it was


def test_the_run_store_resolves_a_unique_prefix_and_lists_newest_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "w.db")
    store.create(RunInfo("20261010-100000-a", "build", "g1", "/p", "m"))
    store.create(RunInfo("20261010-110000-b", "plan", "g2", "/p", "m"))
    store.update("20261010-100000-a", state="paused")
    assert store.resolve("20261010-1100") is not None and store.resolve("20261010-1") is None  # ambiguous
    assert store.resolve("nope") is None
    assert [r.run_id for r in store.list()][0] == "20261010-100000-a"  # updated last
    got = store.get("20261010-100000-a")
    assert got is not None and got.state == "paused"


async def test_a_cancelled_build_resumes_from_its_last_checkpoint_without_repeating_finished_nodes(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    ws = await Workspace.create(fixture_repo, tmp_path / "ws", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    plan = Plan(goal="g", behaviors=["mul multiplies"], files=["src/mod.py"])
    cfg = WorkflowConfig(max_polish_rounds=0)
    db = tmp_path / "workflows.db"
    rec = Recorder()
    first = Hangs([MUL_TEST, MUL_IMPL], at=1)  # write_test returns, write_impl hangs
    async with open_checkpointer(db) as saver:
        graph = build_execution_graph(first, LocalSandbox(), ws, cfg, hints=False, recorder=rec)
        task = asyncio.ensure_future(drive_graph(graph, initial_state(plan), lambda p: "no", thread_id="run1", checkpointer=saver))
        for _ in range(200):
            await asyncio.sleep(0.05)
            if first.calls == 2:
                break
        assert first.calls == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # a new process: fresh saver connection, fresh Recorder, fresh model that only knows the rest of the script
    rec2 = Recorder()
    second = ScriptedModel([MUL_IMPL])
    restored: list[dict[str, Any]] = []

    def on_resume(values: dict[str, Any]) -> None:
        rec2.load(values["recorder"])
        restored.append(values)

    async with open_checkpointer(db) as saver:
        graph = build_execution_graph(second, LocalSandbox(), ws, cfg, hints=False, recorder=rec2)
        final = await drive_graph(
            graph, initial_state(plan), lambda p: "yes", thread_id="run1", checkpointer=saver, resume=True,
            on_resume=on_resume,
        )
    assert len(second.prompts) == 1  # only the cut-off write_impl ran again; write_test was not repeated
    assert final["status"] == "applied" and "def mul" in tree(fixture_repo)["src/mod.py"]
    assert rec2.trace[:3] == ["check_assumptions", "write_test", "run_red"] and rec2.trace[-1] == "finish"
    assert [b.green for b in rec2.behaviors] == [True]
    assert restored and "recorder" in restored[0]
