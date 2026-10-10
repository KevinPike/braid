import asyncio
import io
import json
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.config import HarnessConfig, WorkflowConfig
from harness.sandbox import LocalSandbox
from harness.guard.ollama_client import OllamaClient
from harness.workflow.cli import build_parser, run_build, run_cancellable
from harness.workflow.model import GuardedWorkflowModel
from harness.workflow.plan import Plan
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, ScriptedModel, edit, tree


def test_the_parser_takes_the_documented_flags() -> None:
    args = build_parser().parse_args(["build", "--plan", "p.json", "--model", "m", "--no-hints"])
    assert args.plan == Path("p.json") and args.project == Path(".") and args.model == "m" and args.no_hints


async def test_build_runs_a_plan_prints_progress_writes_the_summary_and_applies_on_yes(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    cfg = HarnessConfig(workflow=WorkflowConfig(
        workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
    ))
    answers = iter(["yes"])
    asked: list[str] = []

    def ask(payload: dict[str, Any]) -> str:
        asked.append(payload["kind"])
        return next(answers)

    out, err = io.StringIO(), io.StringIO()
    model = ScriptedModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    code = await run_build(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=model, model_id="fake:1", sandbox=LocalSandbox(),
        config=cfg, hints=False, ask=ask, out=out, err=err,
    )
    text = out.getvalue()
    assert code == 0 and asked == ["finish"] and err.getvalue() == ""  # not a TTY: no live line
    assert "run_green         ✓ all tests pass" in text and "mul multiplies" in text and "status: applied" in text
    assert tree(fixture_repo)["src/mod.py"] == MUL
    [summary] = (fixture_repo / "runs").glob("*.json")
    data = json.loads(summary.read_text())
    assert data["no_hints"] is True and data["totals"]["green"] == 1 and data["planned"] == 1 and data["trace"][-1] == "finish"
    [log] = (tmp_path / "runs").glob("*.jsonl")
    assert len(log.read_text().splitlines()) == len(data["trace"])


async def test_a_no_hints_run_never_asks_for_anything_but_the_finish_approval(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    cfg = HarnessConfig(workflow=WorkflowConfig(
        workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", max_polish_rounds=0,
        test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
    ))
    asked: list[str] = []

    def ask(payload: dict[str, Any]) -> str:
        asked.append(payload["kind"])
        return "no"

    model = ScriptedModel([MUL_TEST, MUL_IMPL])
    code = await run_build(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=model, model_id="fake:1", sandbox=LocalSandbox(),
        config=cfg, hints=False, ask=ask, out=io.StringIO(), err=io.StringIO(),
    )
    assert code == 0 and asked == ["finish"]
    assert not (fixture_repo / "src" / "mod.py").read_text().count("mul")


def test_plan_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["build"])


class BlockingStream(httpx.AsyncByteStream):
    """An /api/chat response that never ends; records whether the client closed it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await asyncio.Event().wait()
        yield b""

    async def aclose(self) -> None:
        self.closed = True


async def test_cancelling_a_build_closes_the_stream_keeps_the_workspace_and_writes_a_cancelled_summary(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    cfg = HarnessConfig(workflow=WorkflowConfig(workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs"))
    stream = BlockingStream()
    client = OllamaClient("http://ollama.test", http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))))
    client.pin("m", 4096)
    out = io.StringIO()
    task = asyncio.ensure_future(run_build(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=GuardedWorkflowModel(client, "m"), model_id="m",
        sandbox=LocalSandbox(), config=cfg, hints=False, ask=lambda _p: "no", out=out, err=io.StringIO(),
    ))
    await asyncio.wait_for(stream.started.wait(), 60)
    task.cancel()
    assert await task == 130
    assert stream.closed
    [summary] = (fixture_repo / "runs").glob("*.json")
    data = json.loads(summary.read_text())
    assert data["status"] == "cancelled" and Path(data["workspace"]).is_dir()
    assert f"Workspace kept at {data['workspace']}" in out.getvalue()


async def test_sigint_cancels_the_running_task() -> None:
    async def blocked() -> int:
        await asyncio.Event().wait()
        return 0

    runner = asyncio.ensure_future(run_cancellable(blocked()))
    await asyncio.sleep(0.1)
    os.kill(os.getpid(), signal.SIGINT)
    assert await asyncio.wait_for(runner, 5) == 130
