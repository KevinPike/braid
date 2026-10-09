import io
import json
from pathlib import Path
from typing import Any

import pytest

from harness.config import HarnessConfig, WorkflowConfig
from harness.sandbox import LocalSandbox
from harness.workflow.cli import build_parser, run_build
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
