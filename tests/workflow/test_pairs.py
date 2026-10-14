"""Planner x coder (ADR 0009): a Plan records its author, `plans-for` writes Plans, `bench --plans-dir` builds them, and
`report --pairs` tabulates them. No network."""

import argparse
import io
from pathlib import Path
from typing import Any

import pytest

from harness.config import HarnessConfig
from harness.workflow import bench, cli
from harness.workflow.model import WorkflowModel
from harness.workflow.plan import Plan, PlanAuthor
from tests.workflow.test_bench import line


def test_a_plan_keeps_its_author_through_save_and_load_and_the_hash_ignores_it(tmp_path: Path) -> None:
    plan = Plan(goal="g", behaviors=["b"])
    bare = bench.plan_sha256(plan)
    plan.author = PlanAuthor(model="claude-sonnet-5-5", backend="anthropic", effort="low", written_at="2026-10-13T20:00:00")
    plan.save(tmp_path / "p.json")
    again = Plan.load(tmp_path / "p.json")
    assert again.author == plan.author and bench.plan_sha256(again) == bare
    assert Plan.model_validate_json('{"goal": "g", "behaviors": ["b"]}').author is None  # Plans from before


def test_provenance_names_the_planner_only_when_the_plan_has_an_author() -> None:
    kw: dict[str, Any] = {"plan_path": None, "backend": "openrouter", "effort": "low", "max_attempts": 3, "execution": "x", "bench": None}
    plan = Plan(goal="g", behaviors=["b"])
    assert "planner" not in bench.provenance(plan, **kw)
    plan.author = PlanAuthor(model="claude-sonnet-5-5", backend="anthropic", effort="medium")
    prov = bench.provenance(plan, **kw)
    assert prov["planner"] == "claude-sonnet-5-5" and prov["planner_effort"] == "medium"


def test_plans_dir_swaps_each_pairs_plan_and_names_every_missing_one(tmp_path: Path) -> None:
    pairs = tuple(bench.Pair(k, tmp_path / f"{k}-old.json", tmp_path / k) for k in ("alpha", "beta"))
    bset = bench.BenchSet("s", "", pairs)
    (tmp_path / "alpha.json").write_text("{}")
    with pytest.raises(ValueError, match="for: beta"):
        bench.with_plans_dir(bset, tmp_path)
    (tmp_path / "beta.json").write_text("{}")
    swapped = bench.with_plans_dir(bset, tmp_path)
    assert [p.plan.name for p in swapped.pairs] == ["alpha.json", "beta.json"] and swapped.pairs[0].project == pairs[0].project


def test_the_pairs_table_has_a_row_per_planner_and_a_column_per_coder() -> None:
    def graded(label: str, planner: str | None, model: str, kata: str, passed: int, cost: float) -> dict[str, Any]:
        extra: dict[str, Any] = {"planner": planner} if planner else {}
        return line(label, kata, 1, 6, cost=cost, model=model, hidden_passed=passed, hidden_total=10, mutants_killed=3, mutants_total=5, **extra)

    lines = [
        graded("a", "claude-sonnet-5-5", "deepseek/v4", "k1", 9, 0.01), graded("a", "claude-sonnet-5-5", "deepseek/v4", "k2", 9, 0.01),
        graded("b", None, "deepseek/v4", "k1", 7, 0.02), graded("b", None, "deepseek/v4", "k2", 7, 0.02),
        graded("c", "claude-sonnet-5-5", "claude-haiku-5-5", "k1", 8, 0.03), graded("c", "claude-sonnet-5-5", "claude-haiku-5-5", "k2", 8, 0.03),
    ]
    table = bench.pairs_table(lines, ["a", "b", "c", "nope"])
    assert "| planner \\ coder | claude-haiku-5-5 | deepseek/v4 |" in table
    assert "| claude-sonnet-5-5 | 80% · $0.0300 | 90% · $0.0100 |" in table
    assert "| manifest |  | 70% · $0.0200 |" in table
    assert bench.pairs_table(lines, ["nope"]).startswith("no ledger lines labelled")


async def test_plans_for_writes_one_authored_plan_per_kata_and_skips_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("alpha", "beta"):
        Plan(goal=f"goal {k}", behaviors=["b"]).save(tmp_path / f"{k}.json")
    (tmp_path / "katas.toml").write_text(
        f'practice_dir = "{tmp_path}"\n[sets.s]\npairs = [{{kata = "alpha", plan = "alpha.json"}}, {{kata = "beta", plan = "beta.json"}}]\n',
    )
    seen: list[str] = []

    async def fake_run_plan(goal: str, project: Path, *, model: Any, model_id: str, config: Any, ask: Any, on_saved: Any = None, **_: Any) -> int:
        seen.append(goal)
        assert ask({"kind": "review"}) == "approve"  # no human: every review is approved
        if "beta" in goal:
            return 4  # an API error: no Plan
        saved = tmp_path / "saved.json"
        Plan(goal=goal, behaviors=["b"], author=PlanAuthor(model=model_id, backend="openrouter")).save(saved)
        on_saved(saved)
        return 0

    async def fake_with_model(args: argparse.Namespace, config: HarnessConfig, body: Any) -> int:
        fake: WorkflowModel = None  # type: ignore[assignment]
        return int(await body(fake, "z-ai/glm-5.3"))

    monkeypatch.setattr(cli, "run_plan", fake_run_plan)
    monkeypatch.setattr(cli, "with_model", fake_with_model)
    out = tmp_path / "out"
    args = cli.build_parser().parse_args(["plans-for", "s", "--model", "z-ai/glm-5.3", "--out", str(out), "--manifest", str(tmp_path / "katas.toml")])
    assert await cli.live_plans_for(args, HarnessConfig()) == 1  # one kata failed
    assert seen == ["goal alpha", "goal beta"]
    saved = Plan.load(out / "alpha.json")
    assert saved.author is not None and saved.author.model == "z-ai/glm-5.3" and not (out / "beta.json").exists()


def test_report_pairs_prints_the_table(tmp_path: Path) -> None:
    ledger = tmp_path / "bench.jsonl"
    for ln in (line("a", "k1", 1, 6, model="deepseek/v4", planner="claude-sonnet-5-5"),):
        bench.append_ledger(ledger, ln)
    text = io.StringIO()
    args = cli.build_parser().parse_args(["report", "--pairs", "a", "--ledger", str(ledger)])
    assert cli.report(args, HarnessConfig(), text) == 0
    assert "| claude-sonnet-5-5 | 100% green · $0.0100 |" in text.getvalue()
