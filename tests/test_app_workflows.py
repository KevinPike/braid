"""M15 gates: Workflows in the TUI (Textual pilot, scripted models, LocalSandbox, a fake /api/ps and a fake clock)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import pytest
from strands.hooks.events import BeforeToolCallEvent
from textual.pilot import Pilot
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.approval import APPLY_WORKSPACE, ApprovalGate
from harness.chat import AnswerInput, AnswerPrompt, ApprovalPrompt, ContextPanel
from harness.config import HarnessConfig, PathsConfig, ToolsConfig, WorkflowConfig
from harness.debuglog import DebugLog
from harness.guard.watchdog import Watchdog
from harness.hooks import ApprovalRequested, ApprovalResolved, ModelRequestSent, WorkflowInterrupted, WorkflowNodeFinished
from harness.inspector import RunInspector
from harness.sandbox import LocalSandbox
from harness.status import StatusBar
from harness.workflow.model import WorkflowModel
from harness.workflow.plan import Assumption, Plan
from harness.workflow.progress import final_report
from harness.workflow.records import read_log
from harness.workflow.runs import RunInfo, RunStore
from harness.workflow_pane import WorkflowDeps, parse_flags, shorten_command
from tests.guard.fakes import GIB, FakeBackend
from tests.workflow.test_checkpoint import Hangs
from tests.workflow.test_execute import MOD, MUL_IMPL, MUL_TEST, NEG, NEG_TEST, ScriptedModel, edit, make_test, tree
from tests.workflow.test_plan_loop import scripted

WRONG = edit("src/mod.py", MOD)  # an implementation that leaves the new test failing (import error: pytest exit 2)
WRONG_MUL = edit("src/mod.py", MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return 0\n")  # a failing assertion: pytest exit 1


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Gated:
    """A scripted model whose call number ``at`` waits until ``release`` is set."""

    def __init__(self, replies: list[Any], at: int) -> None:
        self.inner = ScriptedModel(replies)
        self.at, self.calls = at, 0
        self.release = asyncio.Event()

    async def complete(self, system: str, prompt: str, schema: Any) -> Any:
        self.calls += 1
        if self.calls - 1 == self.at:
            await self.release.wait()
        return await self.inner.complete(system, prompt, schema)


class Boom:
    async def complete(self, system: str, prompt: str, schema: Any) -> Any:
        raise RuntimeError("the model fell over")


async def idle_reply(prompt: str) -> AsyncIterator[str]:
    yield f"reply to {prompt}"


def make_config(tmp_path: Path, repo: Path, venv: Path, **wf: Any) -> HarnessConfig:
    settings: dict[str, Any] = {
        "workspaces_dir": tmp_path / "ws", "runs_dir": tmp_path / "runs", "plans_dir": tmp_path / "plans",
        "checkpoints_db": tmp_path / "workflows.db", "max_polish_rounds": 0, "model": "fake:1", "num_ctx": 4096,
        "test_command": f"{venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
    }
    settings.update(wf)
    return HarnessConfig(tools=ToolsConfig(root=repo), paths=PathsConfig(data_dir=tmp_path / "data"), workflow=WorkflowConfig(**settings))


def save_plan(tmp_path: Path, behaviors: list[str] | None = None, name: str = "mulplan", assumptions: list[Assumption] | None = None) -> Path:
    path = tmp_path / "plans" / f"{name}.json"
    Plan(goal="mul helper", behaviors=behaviors or ["mul multiplies"], files=["src/mod.py"], assumptions=assumptions or []).save(path)
    return path


def make_app(
    config: HarnessConfig, model: Any, *, replier: Any = idle_reply, clock: Clock | None = None, **extra: Any,
) -> HarnessApp:
    holder = model if isinstance(model, dict) else {"model": model}
    deps = WorkflowDeps(
        model_factory=lambda _id: cast(WorkflowModel, holder["model"]), sandbox=LocalSandbox(), clock=clock or time.monotonic,
        refresh_s=0.05, **{k: v for k, v in extra.items() if k in ("backend", "chat_model", "chat_num_ctx", "gpu_limit", "env")},
    )
    rest = {k: v for k, v in extra.items() if k not in ("backend", "chat_model", "chat_num_ctx", "gpu_limit", "env")}
    return HarnessApp(config, replier=replier, workflows=deps, **rest)


async def until(pilot: Pilot[None], predicate: Callable[[], bool], timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for the condition")
        await pilot.pause(0.05)


async def submit(app: HarnessApp, text: str) -> None:
    box = app.query_one("#prompt", Input)
    box.value = text
    await box.action_submit()


async def answer(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    prompt = app.query_one(AnswerPrompt)
    field = prompt.query_one(AnswerInput)
    field.value = text
    await field.action_submit()
    await until(pilot, lambda: not prompt.is_attached)


def texts(app: HarnessApp, css_class: str) -> list[str]:
    return [str(w.render()) for w in app.query(Static) if w.has_class(css_class)]


def records(app: HarnessApp) -> list[str]:
    return texts(app, "record")


async def finished(app: HarnessApp, pilot: Pilot[None]) -> None:
    await until(pilot, lambda: not app.workflows.running and not app.generating)


def run_ids(app: HarnessApp) -> list[str]:
    return [r.run_id for r in app.workflows.store.list()]


# --- gate 1 -------------------------------------------------------------------------------------------------------


async def test_build_runs_to_a_stuck_interrupt_and_a_prompt_submitted_meanwhile_stays_queued_until_the_widget_is_answered(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    model = ScriptedModel([MUL_TEST, WRONG, WRONG, WRONG, MUL_IMPL])
    ran: list[str] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        yield "ok"

    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), model, replier=replier)
    interrupts: list[str] = []
    app.hooks.add_callback(WorkflowInterrupted, lambda e: interrupts.append(e.kind))
    async with app.run_test() as pilot:
        await submit(app, "/build")
        await until(pilot, lambda: bool(app.query(AnswerPrompt)))
        assert "stuck on: mul multiplies" in str(app.query_one(AnswerPrompt).query_one(Static).render())
        await submit(app, "hello")
        await pilot.pause(0.3)
        assert ran == [] and app.queue.pending == ("hello",)
        assert app.query_one("#prompt").border_title == "1 queued · held by workflow question"
        stuck = app.query_one(AnswerPrompt)
        await answer(app, pilot, "multiply them")
        await until(pilot, lambda: bool(app.query(AnswerPrompt)) and app.query_one(AnswerPrompt) is not stuck)  # sign-off
        assert "multiply them" in model.prompts[-1] and ran == []
        await answer(app, pilot, "yes")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        assert ran == [] and app.queue.pending == ("hello",)
        await pilot.press("n")
        await until(pilot, lambda: ran == ["hello"])
        await finished(app, pilot)
        recs = records(app)
    assert interrupts == ["stuck", "sign_off", "finish"]
    assert recs[:2] == ["✓ stuck on: mul multiplies? multiply them", "✓ sign-off? yes"]


# --- gate 2 -------------------------------------------------------------------------------------------------------


async def test_the_finish_approval_emits_both_approval_events_and_n_leaves_the_project_unchanged(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    before = tree(fixture_repo)
    events: list[Any] = []
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([MUL_TEST, MUL_IMPL]))
    app.hooks.add_callback(ApprovalRequested, lambda e: events.append(e))
    app.hooks.add_callback(ApprovalResolved, lambda e: events.append(e))
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        prompt = app.query_one(ApprovalPrompt)
        assert prompt.request.tool == APPLY_WORKSPACE and "def mul" in str(prompt.render())  # the diff is on screen
        assert prompt.parent is app.query_one("#chat") and app.approval_pending
        await pilot.press("n")
        await finished(app, pilot)
        recs = records(app)
    assert [type(e).__name__ for e in events] == ["ApprovalRequested", "ApprovalResolved"]
    assert events[1].approved is False and events[0].request is events[1].request
    assert tree(fixture_repo) == before
    assert any("apply the Workspace diff" in r and r.endswith("denied") for r in recs)
    [info] = app.workflows.store.list()
    assert (info.state, info.outcome) == ("finished", "kept")


async def test_y_at_the_finish_approval_applies_the_diff(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    save_plan(tmp_path)
    resolved: list[bool] = []
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([MUL_TEST, MUL_IMPL]))
    app.hooks.add_callback(ApprovalResolved, lambda e: resolved.append(e.approved))
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("y")
        await finished(app, pilot)
    assert resolved == [True] and "def mul" in tree(fixture_repo)["src/mod.py"]
    assert app.workflows.store.list()[0].outcome == "applied"


# --- gate 3 -------------------------------------------------------------------------------------------------------


async def test_ctrl_c_during_a_node_then_resume_finishes_without_rerunning_completed_nodes_and_the_table_counts_earlier_behaviors(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path, ["mul multiplies", "neg negates"])
    first = Hangs([MUL_TEST, MUL_IMPL, NEG_TEST], at=3)  # Behavior 1 is done; Behavior 2's write_impl hangs
    holder: dict[str, Any] = {"model": first}
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), holder)
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: first.calls == 4)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.workflows.running)
        [info] = app.workflows.store.list()
        assert info.state == "paused" and not Path(info.workspace).joinpath("src", "new.py").exists()
        assert tree(fixture_repo)["src/mod.py"] == MOD  # nothing was applied
        assert any("paused at its last checkpoint" in t for t in texts(app, "note"))
        second = ScriptedModel([edit("src/mod.py", NEG)])
        holder["model"] = second
        await submit(app, "")  # Enter on an empty prompt resumes the paused queue
        await submit(app, f"/build --resume {info.run_id[:12]}")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        assert len(second.prompts) == 1  # only the cut-off write_impl ran again; nothing finished was repeated
        await pilot.press("y")
        await finished(app, pilot)
    assert "def neg" in tree(fixture_repo)["src/mod.py"]
    report = app.workflows.last_report
    assert report.count("green") >= 2 and "mul multiplies" in report and "neg negates" in report
    assert report == final_report(app.workflows.recorder, app.workflows.last_total) + "\n\nstatus: applied; polish not run"
    assert [(b.behavior, b.green) for b in app.workflows.recorder.behaviors] == [("mul multiplies", True), ("neg negates", True)]
    assert app.workflows.recorder.behaviors[0].attempts == 1  # counted before the cancel
    assert app.workflows.store.list()[0].state == "finished"


async def test_ctrl_c_kills_the_sandbox_process_group_of_the_running_command_and_applies_nothing(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    pidfile = tmp_path / "probe.pid"
    save_plan(tmp_path, assumptions=[Assumption(text="slow probe", probe=f"echo $$ > {pidfile}; sleep 120")])
    before = tree(fixture_repo)
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([]))
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: pidfile.exists() and pidfile.read_text().strip().isdigit())
        pid = int(pidfile.read_text())
        os.kill(pid, 0)  # alive
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.workflows.running)

        def gone() -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            return False

        await until(pilot, gone, timeout=10.0)
        assert app.workflows.store.list()[0].state == "paused" and tree(fixture_repo) == before


# --- gate 4 -------------------------------------------------------------------------------------------------------


async def test_killing_the_app_mid_run_and_resuming_in_a_new_one_continues_from_the_checkpoint(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path, ["mul multiplies", "neg negates"])
    config = make_config(tmp_path, fixture_repo, fixture_venv)
    first = Hangs([MUL_TEST, MUL_IMPL, NEG_TEST], at=3)
    app = make_app(config, first)
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: first.calls == 4)
        run_id = run_ids(app)[0]
    # the app is gone. A real kill leaves the run marked running, with its checkpoint and Workspace on disk
    RunStore(config.workflow.checkpoints_db).update(run_id, state="running")
    second = ScriptedModel([edit("src/mod.py", NEG)])
    app2 = make_app(config, second)
    async with app2.run_test() as pilot:
        await submit(app2, "/runs")
        await until(pilot, lambda: any(run_id in t and "killed" in t for t in texts(app2, "note")))
        await submit(app2, f"/build --resume {run_id}")
        await until(pilot, lambda: bool(app2.query(ApprovalPrompt)))
        await pilot.press("y")
        await finished(app2, pilot)
    assert len(second.prompts) == 1
    assert "def neg" in tree(fixture_repo)["src/mod.py"]
    assert [b.green for b in app2.workflows.recorder.behaviors] == [True, True]
    assert app2.workflows.recorder.trace[0] == "check_assumptions" and app2.workflows.recorder.trace[-1] == "finish"


# --- gate 5 -------------------------------------------------------------------------------------------------------


async def test_every_node_update_emits_workflow_node_finished(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    save_plan(tmp_path)
    events: list[WorkflowNodeFinished] = []
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([MUL_TEST, WRONG, MUL_IMPL]))
    app.hooks.add_callback(WorkflowNodeFinished, lambda e: events.append(e))
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("n")
        await finished(app, pilot)
    run_id = run_ids(app)[0]
    logged = read_log(tmp_path / "runs" / f"{run_id}.jsonl")
    assert [e.node for e in events] == [r["node"] for r in logged] and len(events) >= 8
    assert {e.run_id for e in events} == {run_id}
    assert all(e.duration_s >= 0 for e in events)
    assert [e.attempt for e in events if e.node == "write_impl"] == [1, 2]  # the attempt number comes from the node's log line


# --- gate 6 -------------------------------------------------------------------------------------------------------


async def test_a_stuck_interrupt_and_a_chat_shell_approval_are_inline_widgets_that_collapse_to_a_record_and_pytest_shows_inline(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([MUL_TEST, WRONG_MUL, WRONG_MUL, WRONG_MUL]))
    gate = ApprovalGate(app.ask_approval, hooks=app.hooks)
    async with app.run_test() as pilot:
        assert not app.query(ApprovalPrompt) and not app.query(AnswerPrompt)  # nothing fixed in compose()
        chat = app.query_one("#chat")
        await submit(app, "/build")
        await until(pilot, lambda: bool(app.query(AnswerPrompt)))
        stuck = app.query_one(AnswerPrompt)
        assert stuck.parent is chat
        pytest_lines = [t for t in texts(app, "command")]
        assert any(t.startswith("$ pytest") and "exit 1" in t and t.rstrip().endswith("s") for t in pytest_lines), pytest_lines
        assert any(re.fullmatch(r"\$ pytest.* exit 1  \d+\.\ds", t) for t in pytest_lines)
        await answer(app, pilot, "abort")
        assert not stuck.is_attached and records(app) == ["✓ stuck on: mul multiplies? abort"]
        await finished(app, pilot)
        # the chat agent's shell tool goes through the same inline path
        event = BeforeToolCallEvent(agent=cast(Any, None), selected_tool=None,
                                    tool_use={"toolUseId": "1", "name": "shell", "input": {"command": "ls"}}, invocation_state={})
        task = asyncio.create_task(gate.before_tool(event))
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        shell = app.query_one(ApprovalPrompt)
        assert shell.parent is chat and "command: ls" in str(shell.render())
        await pilot.press("y")
        await task
        await until(pilot, lambda: not shell.is_attached)
        assert records(app)[-1] == "✓ shell: ls? approved" and not event.cancel_tool


# --- gate 7 -------------------------------------------------------------------------------------------------------


async def test_timing_the_live_line_updates_in_place_a_finished_node_shows_its_duration_the_clock_runs_and_the_table_matches(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    clock = Clock()
    model = Gated([MUL_TEST, MUL_IMPL], at=1)  # write_impl waits
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), model, clock=clock)
    async with app.run_test() as pilot:
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: model.calls == 2)
        await until(pilot, lambda: len(app.query(".live")) == 1)
        live = texts(app, "live")
        assert len(live) == 1 and "write_impl" in live[0] and "Behavior 1/1" in live[0] and "0s" in live[0]
        clock.now += 7
        await until(pilot, lambda: "7s" in "".join(texts(app, "live")))
        assert len(texts(app, "live")) == 1  # the same line, updated in place
        clock.now += 5
        await until(pilot, lambda: "12s" in "".join(texts(app, "live")))
        assert len(texts(app, "live")) == 1
        bar = str(app.query_one(StatusBar).render())
        assert "workflow 0:12" in bar
        done = [t for t in texts(app, "note") if "write_test" in t and "wrote" in t]
        assert len(done) == 1 and "tests/test_behavior_1.py" in done[0]
        assert re.search(r"\d+\.\ds  \(model \d+\.\ds\)$", done[0])  # result, duration and model latency
        model.release.set()
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        clock.now += 3
        await pilot.press("n")
        await finished(app, pilot)
        assert "workflow" not in str(app.query_one(StatusBar).render())
        assert not app.query(".live")
        table = [t for t in texts(app, "table")]
    pane = app.workflows
    assert pane.last_total == 15.0  # 7 + 5 + 3 on the injected clock
    assert table == [pane.last_report]
    assert pane.last_report.startswith(final_report(pane.recorder, 15.0))
    assert "total wall time 15.0s" in pane.last_report and "mul multiplies" in pane.last_report and "green" in pane.last_report


# --- gate 8 -------------------------------------------------------------------------------------------------------


async def test_the_status_bar_renders_the_active_model_first_a_helper_after_it_and_a_total_with_both() -> None:
    backend = FakeBackend(models={"tev1:0.8b": (int(0.8 * GIB), 0.0), "gemma4:12b": (9 * GIB, 0.0)})
    for name in ("tev1:0.8b", "gemma4:12b"):  # the helper loaded first, so ps lists it first
        backend.pin(name, 4096)
        await backend.warm(name)
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.05, memory_level=lambda: 1)
    wd.mark_helper("tev1:0.8b")
    app = HarnessApp(HarnessConfig(), replier=idle_reply, watchdog=wd)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        text = str(app.query_one(StatusBar).render())
    assert text.index("gemma4:12b") < text.index("+ tev1:0.8b 0.8 GiB")
    assert "9.0 GiB" in text and "total 9.8 GiB" in text
    assert [m.name for m in wd.state.models] == ["gemma4:12b", "tev1:0.8b"]


# --- gate 9 and 10 ----------------------------------------------------------------------------------------------


CHAT, WORKFLOW = "gemma4:e4b", "wf:model"


async def swap_app(
    tmp_path: Path, repo: Path, venv: Path, model: Any, *, log: DebugLog | None = None,
) -> tuple[HarnessApp, FakeBackend]:
    backend = FakeBackend(models={CHAT: (3 * GIB, 50_000.0), WORKFLOW: (4 * GIB, 50_000.0), "tev1:0.8b": (GIB, 1000.0)})
    for name, ctx in ((CHAT, 8192), ("tev1:0.8b", 2048)):
        backend.pin(name, ctx)
        await backend.warm(name)
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.02, memory_level=lambda: 1)
    wd.mark_helper("tev1:0.8b")
    wd.set_active(CHAT)
    app = make_app(
        make_config(tmp_path, repo, venv, model=WORKFLOW), model, watchdog=wd, debug_log=log,
        backend=backend, chat_model=CHAT, chat_num_ctx=8192, gpu_limit=lambda: 16 * GIB, env={"OLLAMA_NUM_PARALLEL": "1"},
    )
    return app, backend


def assert_chat_restored(app: HarnessApp, backend: FakeBackend) -> None:
    assert backend.loaded.get(CHAT) == 8192 and WORKFLOW not in backend.loaded
    assert backend.load_log.index((WORKFLOW, 4096)) < len(backend.load_log) - 1 and backend.load_log[-1] == (CHAT, 8192)


async def shows_chat_model(app: HarnessApp, pilot: Pilot[None]) -> None:
    def ok() -> bool:
        bar = str(app.query_one(StatusBar).render())
        return bar.startswith(CHAT) and WORKFLOW not in bar and f"loaded: {CHAT}" in str(app.query_one(ContextPanel).render())

    await until(pilot, ok, timeout=1.0)  # within a poll or a few


async def test_swap_a_completed_run_ends_with_the_chat_model_preflighted_again(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    model = Gated([MUL_TEST, MUL_IMPL], at=1)
    app, backend = await swap_app(tmp_path, fixture_repo, fixture_venv, model)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: model.calls == 2)
        assert backend.loaded.get(WORKFLOW) == 4096 and CHAT not in backend.loaded  # the chat model was unloaded
        await until(pilot, lambda: str(app.query_one(StatusBar).render()).startswith(WORKFLOW))
        assert "workflow 0:" in str(app.query_one(StatusBar).render()) and "+ tev1:0.8b" in str(app.query_one(StatusBar).render())
        assert f"loaded: {WORKFLOW}" in str(app.query_one(ContextPanel).render())
        model.release.set()
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("n")
        await finished(app, pilot)
        await shows_chat_model(app, pilot)
    assert_chat_restored(app, backend)


async def test_an_api_model_skips_preflight_and_the_swap_so_the_chat_model_stays_loaded(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    model = Gated([MUL_TEST, MUL_IMPL], at=1)
    app, backend = await swap_app(tmp_path, fixture_repo, fixture_venv, model)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        loads = list(backend.load_log)
        await submit(app, "/build --no-hints --model claude-haiku-5-5")
        await until(pilot, lambda: model.calls == 2)
        assert backend.loaded.get(CHAT) == 8192 and backend.load_log == loads  # nothing unloaded, nothing preflighted
        model.release.set()
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("n")
        await finished(app, pilot)
        await shows_chat_model(app, pilot)
    assert backend.loaded.get(CHAT) == 8192 and backend.load_log == loads


async def test_swap_a_cancelled_run_ends_with_the_chat_model_preflighted_again(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    hang = Hangs([MUL_TEST], at=0)
    app, backend = await swap_app(tmp_path, fixture_repo, fixture_venv, hang)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: hang.calls == 1)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.workflows.running)
        await shows_chat_model(app, pilot)
    assert_chat_restored(app, backend)
    assert app.workflows.store.list()[0].state == "paused"


async def test_swap_a_failed_run_ends_with_the_chat_model_preflighted_again(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    app, backend = await swap_app(tmp_path, fixture_repo, fixture_venv, Boom())
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: app.workflows.run_id is not None)
        await finished(app, pilot)
        await shows_chat_model(app, pilot)
        assert any("failed: the model fell over" in t for t in texts(app, "note"))
    assert_chat_restored(app, backend)
    [info] = app.workflows.store.list()
    assert info.state == "failed" and "fell over" in info.outcome


async def test_no_empty_frame_across_a_swap_the_loading_state_shows_and_polls_and_renders_are_logged(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    log = DebugLog(tmp_path / "status-debug.log")
    model = Gated([MUL_TEST, MUL_IMPL], at=1)
    app, backend = await swap_app(tmp_path, fixture_repo, fixture_venv, model, log=log)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        bar = app.query_one(StatusBar)
        bar.history.clear()
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: model.calls == 2)
        await pilot.pause(0.2)  # several polls while the workflow model runs
        model.release.set()
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("n")
        await finished(app, pilot)
        await shows_chat_model(app, pilot)
        history = list(bar.history)
    assert history and not [t for t in history if t.startswith("no model loaded") or not t.strip()]
    loading = [t for t in history if t.startswith("loading")]
    assert f"loading {WORKFLOW}…" in loading and f"loading {CHAT}…" in loading
    lines = log.lines()
    polls = [ln for ln in lines if " poll: " in ln]
    rendered = [ln.split(" render: ", 1)[1] for ln in lines if " render: " in ln]
    assert len(polls) >= 4 and any(WORKFLOW in ln for ln in polls) and all(ln[:4].isdigit() for ln in lines)  # timestamped
    assert len(rendered) >= 4 and not [r for r in rendered if r.startswith("no model loaded")]
    assert any(history[i : i + len(rendered)] == rendered for i in range(len(history) - len(rendered) + 1))  # in order, none missing


# --- gate 11 ------------------------------------------------------------------------------------------------------


def write_fixture_log(runs: Path, run_id: str) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    rows = [
        {"node": "write_test", "ts": "2026-10-10T10:00:00.000+00:00", "behavior": 0, "attempt": 1, "node_seconds": 4.2, "model_seconds": 4.0,
         "prompt": "PROMPT-ONE write a test for mul", "reply": {"edits": [{"path": "tests/test_b.py", "content": "x"}], "rationale": "RATIONALE-ONE"},
         "written": ["tests/test_b.py"], "refused": []},
        {"node": "run_red", "ts": "2026-10-10T10:00:05.000+00:00", "behavior": 0, "attempt": 1, "verdict": "red", "exit_code": 1,
         "timed_out": False, "seconds": 0.3, "output_tail": "TAIL-TWO FAILED test_b"},
    ]
    (runs / f"{run_id}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


async def test_runs_with_an_id_opens_the_run_log_in_the_inspector_and_workflow_calls_stay_out_of_the_calls_table(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    config = make_config(tmp_path, fixture_repo, fixture_venv)
    run_id = "20261010-100000-fake_1"
    write_fixture_log(tmp_path / "runs", run_id)
    app = make_app(config, ScriptedModel([]))
    async with app.run_test() as pilot:
        app.workflows.store.create(RunInfo(run_id, "build", "mul helper", str(fixture_repo), "fake:1", state="finished", outcome="applied"))
        await submit(app, "/runs")
        await until(pilot, lambda: any(run_id in t and "finished" in t and "applied" in t for t in texts(app, "note")))
        await submit(app, "/runs 20261010")
        await until(pilot, lambda: isinstance(app.screen, RunInspector))
        detail = app.screen.query_one("#run-detail", Static)
        first = str(detail.render())
        for part in ("write_test", "PROMPT-ONE write a test for mul", "RATIONALE-ONE", "tests/test_b.py", "attempt 1"):
            assert part in first
        await pilot.press("down")
        await pilot.pause(0.1)
        second = str(detail.render())
        for part in ("run_red", "exit_code", "1", "TAIL-TWO FAILED test_b", "red"):
            assert part in second
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, RunInspector))
        assert app.store.calls(app.session_id) == []  # nothing from the run in the session's calls


async def test_a_workflow_model_request_is_not_recorded_as_a_session_call(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    save_plan(tmp_path)
    hang = Hangs([MUL_TEST], at=0)
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), hang)
    async with app.run_test() as pilot:
        app.hooks.publish(ModelRequestSent("m", {"options": {"num_ctx": 1}}))
        assert len(app.store.calls(app.session_id)) == 1  # a chat request is recorded
        await submit(app, "/build --no-hints")
        await until(pilot, lambda: hang.calls == 1)
        app.hooks.publish(ModelRequestSent("fake:1", {"options": {"num_ctx": 4096}}))  # what the guard client publishes mid-run
        assert len(app.store.calls(app.session_id)) == 1
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.workflows.running)


# --- /plan and /runs --------------------------------------------------------------------------------------------


async def test_plan_runs_the_plan_loop_in_the_chat_asks_for_review_inline_and_saves_a_plan_that_build_picks_up(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    schema_model = scripted()
    holder: dict[str, Any] = {"model": schema_model}
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), holder)
    async with app.run_test() as pilot:
        await submit(app, "/plan add a mul helper")
        await until(pilot, lambda: bool(app.query(AnswerPrompt)))
        review = str(app.query_one(AnswerPrompt).query_one(Static).render())
        assert "round 1/3" in review and "mul multiplies two ints" in review and "[spacing] fine" in review
        assert any("round 1/3" in t or "critique" in t for t in texts(app, "note"))
        await answer(app, pilot, "approve")
        await finished(app, pilot)
        saved = sorted((tmp_path / "plans").glob("*.json"))
        assert len(saved) == 1 and Plan.load(saved[0]).goal == "add a mul helper"
        assert any(t.startswith("✓ review of the Plan, round 1/3? approve") for t in records(app))
        assert any(f"plan saved: {saved[0]}" in t for t in texts(app, "table"))
        holder["model"] = ScriptedModel([MUL_TEST, MUL_IMPL, make_test(2, "assert mul(0, 5) == 0")])  # the Plan has two Behaviors
        await submit(app, "/build --no-hints")  # the most recent Plan
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        await pilot.press("n")
        await finished(app, pilot)
    kinds = {r.kind for r in app.workflows.store.list()}
    assert kinds == {"plan", "build"}


async def test_runs_lists_runs_with_their_state_and_unknown_ids_are_reported(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    app = make_app(make_config(tmp_path, fixture_repo, fixture_venv), ScriptedModel([]))
    async with app.run_test() as pilot:
        await submit(app, "/runs")
        await until(pilot, lambda: any("no workflow runs yet" in t for t in texts(app, "note")))
        app.workflows.store.create(RunInfo("r-paused", "build", "goal a", "/p", "m", state="paused"))
        await submit(app, "/runs")
        await until(pilot, lambda: any("r-paused" in t and "paused" in t and "/build --resume r-paused" in t for t in texts(app, "note")))
        await submit(app, "/runs nothing")
        await until(pilot, lambda: any("no single run matches 'nothing'" in t for t in texts(app, "note")))


def test_flags_come_first_and_the_rest_is_the_goal_and_a_pytest_command_is_shown_without_the_interpreter_path() -> None:
    assert parse_flags("--model m --no-hints add a --flag helper") == ({"--model": "m", "--no-hints": "1"}, "add a --flag helper")
    assert parse_flags("--resume 2026") == ({"--resume": "2026"}, "")
    with pytest.raises(ValueError, match="--nope"):
        parse_flags("--nope x")
    with pytest.raises(ValueError, match="--model"):
        parse_flags("--model")
    assert shorten_command("/tmp/v/.venv/bin/python -m pytest -q tests/test_x.py") == "pytest -q tests/test_x.py"
    assert shorten_command("test -f pyproject.toml") == "test -f pyproject.toml"
    assert len(shorten_command("echo " + "x" * 300)) == 90
