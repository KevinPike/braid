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
from harness.workflow.cli import build_parser, run_build, run_cancellable, run_plan
from harness.workflow.model import GuardedWorkflowModel, WorkflowModelError
from harness.workflow.plan import Plan
from tests.workflow.test_plan_loop import draft, scripted
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
    nodes = [json.loads(line)["node"] for line in log.read_text().splitlines()]
    assert len([n for n in nodes if n != "route"]) == len(data["trace"])
    assert data["routes"] == {"code": nodes.count("route"), "model": 0, "human": 0}  # a no-hints run: code answers


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


# --- plan (M14) ---------------------------------------------------------------------------------------------------

def plan_config(tmp_path: Path, fixture_venv: Path) -> HarnessConfig:
    return HarnessConfig(workflow=WorkflowConfig(
        workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", plans_dir=tmp_path / "plans", max_polish_rounds=0,
        test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
    ))


def test_the_plan_parser_takes_a_goal_and_the_shared_flags() -> None:
    args = build_parser().parse_args(["plan", "add mul", "--project", "p", "--model", "m"])
    assert args.command == "plan" and args.goal == "add mul" and args.project == Path("p") and args.model == "m"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan"])


async def test_plan_prints_progress_the_review_and_the_saved_path_and_build_consumes_that_plan(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    cfg = plan_config(tmp_path, fixture_venv)
    planner = scripted(PlanDraft=[draft(behaviors=["mul multiplies two ints"])])
    reviews: list[dict[str, Any]] = []
    answers = iter(["make it small", "approve"])

    def review(payload: dict[str, Any]) -> str:
        reviews.append(payload)
        return next(answers)

    out, err = io.StringIO(), io.StringIO()
    code = await run_plan(
        "add a mul helper", fixture_repo, model=planner, model_id="fake:1", config=cfg, ask=review, out=out, err=err,
    )
    text = out.getvalue()
    [saved] = (tmp_path / "plans").glob("*.json")
    assert code == 0 and err.getvalue() == "" and saved.name.endswith("-add-a-mul-helper.json")
    assert [r["kind"] for r in reviews] == ["review", "review"]
    assert "HUMAN: review" in text and "approve | research:" in text and "[spacing] fine" in text and "round 1/3" in text
    assert "  research          1 findings" in text and "  critique          [spacing] 1 points" in text
    assert "# add a mul helper" in text and f"plan saved: {saved}" in text and f"harness-workflow build --plan {saved}" in text
    [log] = (tmp_path / "runs").glob("plan-*.jsonl")
    assert [json.loads(line)["node"] for line in log.read_text().splitlines()].count("critique") == 10

    # the saved Plan feeds `build` unchanged
    build_out = io.StringIO()
    code = await run_build(
        Plan.load(saved), fixture_repo, model=ScriptedModel([MUL_TEST, MUL_IMPL]), model_id="fake:1", sandbox=LocalSandbox(),
        config=cfg, hints=False, ask=lambda _p: "yes", out=build_out, err=io.StringIO(),
    )
    assert code == 0 and "status: applied" in build_out.getvalue()
    assert tree(fixture_repo)["src/mod.py"] == MUL


async def test_plan_shows_the_live_line_on_a_tty_with_the_round(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    err = Tty()
    code = await run_plan(
        "g", fixture_repo, model=scripted(), model_id="m", config=plan_config(tmp_path, fixture_venv), ask=lambda _p: "approve",
        out=io.StringIO(), err=err, tty=True,
    )
    assert code == 0 and "research · round 1/3" in err.getvalue()


async def test_a_model_failure_ends_the_plan_run_with_exit_1_and_saves_nothing(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    class Broken:
        async def complete(self, system: str, prompt: str, schema: Any) -> Any:
            raise WorkflowModelError("bad json twice")

    err = io.StringIO()
    code = await run_plan(
        "g", fixture_repo, model=Broken(), model_id="m", config=plan_config(tmp_path, fixture_venv), ask=lambda _p: "approve",
        out=io.StringIO(), err=err,
    )
    assert code == 1 and "bad json twice" in err.getvalue() and not (tmp_path / "plans").exists()


async def test_cancelling_a_plan_closes_the_stream_exits_130_and_saves_no_plan(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    stream = BlockingStream()
    client = OllamaClient("http://ollama.test", http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))))
    client.pin("m", 4096)
    out = io.StringIO()
    task = asyncio.ensure_future(run_plan(
        "g", fixture_repo, model=GuardedWorkflowModel(client, "m"), model_id="m", config=plan_config(tmp_path, fixture_venv),
        ask=lambda _p: "approve", out=out, err=io.StringIO(),
    ))
    await asyncio.wait_for(stream.started.wait(), 60)
    task.cancel()
    assert await task == 130
    assert stream.closed and "no Plan saved" in out.getvalue() and not (tmp_path / "plans").exists()
