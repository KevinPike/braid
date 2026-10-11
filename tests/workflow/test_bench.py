"""F2 gates: provenance in summaries, the bench runner and its ledger, the bench-wide spend stop, and the comparison."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from harness.sandbox import LocalSandbox
from harness.workflow import bench
from harness.workflow.backend import Choice
from harness.workflow.cli import SPEND_CAP_EXIT, build_once, run_bench
from harness.workflow.execute import EXECUTION, execution_variant
from harness.workflow.plan import Plan
from tests.conftest import make_fixture_repo
from tests.workflow.test_api_runs import CostlyModel, cfg
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, ScriptedModel, edit

PROVENANCE = {"harness_commit", "harness_dirty", "plan_path", "plan_sha256", "backend", "model", "effort",
              "max_attempts", "execution", "bench"}


def replies() -> list[Any]:
    return [MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)]


# --- gate 1: provenance ---------------------------------------------------------------------------------------------

async def test_a_summary_carries_its_provenance(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    plan = Plan(goal="g", behaviors=["mul multiplies"])
    path = tmp_path / "plan.json"
    plan.save(path)
    result = await build_once(
        plan, fixture_repo, model=ScriptedModel(replies()), model_id="claude-haiku-5-5", sandbox=LocalSandbox(),
        config=cfg(tmp_path, fixture_venv, effort="low"), hints=False, ask=lambda _p: "no", out=io.StringIO(), err=io.StringIO(),
        plan_path=path, bench_label="lbl",
    )
    assert result.summary is not None
    summary = json.loads(result.summary.read_text())
    assert PROVENANCE <= summary.keys()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=bench.HARNESS_ROOT, capture_output=True, text=True).stdout.strip()
    assert summary["harness_commit"] == head and isinstance(summary["harness_dirty"], bool)
    assert summary["plan_path"] == str(path) and summary["plan_sha256"] == bench.plan_sha256(plan)
    assert (summary["backend"], summary["model"], summary["effort"]) == ("anthropic", "claude-haiku-5-5", "low")
    assert (summary["max_attempts"], summary["execution"], summary["bench"]) == (3, EXECUTION, "lbl")


def test_the_execution_variant_names_the_switches() -> None:
    from harness.config import WorkflowConfig

    assert execution_variant(WorkflowConfig()) == EXECUTION
    assert execution_variant(WorkflowConfig(check_tests_at_red=False, show_invariants=False)) == f"{EXECUTION}/no-red-check/hidden-rules"


def test_the_plan_hash_changes_with_the_plan_and_not_with_its_formatting(tmp_path: Path) -> None:
    plan = Plan(goal="g", behaviors=["one", "two"])
    path = tmp_path / "p.json"
    path.write_text(json.dumps(plan.model_dump(), indent=8))
    assert bench.plan_sha256(Plan.load(path)) == bench.plan_sha256(plan)
    assert bench.plan_sha256(Plan(goal="g", behaviors=["one", "two!"])) != bench.plan_sha256(plan)


def test_an_ollama_build_records_no_effort(tmp_path: Path) -> None:
    fields = bench.provenance(Plan(goal="g", behaviors=["b"]), plan_path=None, backend="ollama", effort=None,
                              max_attempts=2, execution=EXECUTION, bench=None)
    assert fields["effort"] is None and fields["bench"] is None and fields["plan_path"] is None


# --- gates 2 and 3: the bench runner -------------------------------------------------------------------------------

def two_katas(tmp_path: Path) -> bench.BenchSet:
    pairs = []
    for name in ("alpha", "beta"):
        repo = make_fixture_repo(tmp_path / "katas" / name)
        plan_path = tmp_path / "plans" / f"{name}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        Plan(goal=f"{name} mul", behaviors=["mul multiplies"]).save(plan_path)
        pairs.append(bench.Pair(name, plan_path, repo))
    return bench.BenchSet("two", "", tuple(pairs))


def worktrees(repo: Path) -> list[str]:
    out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.startswith("worktree ")]


async def test_bench_runs_every_pair_n_times_into_the_ledger_and_leaves_no_workspace(
    fixture_venv: Path, tmp_path: Path,
) -> None:
    bench_set = two_katas(tmp_path)
    ledger = tmp_path / "bench.jsonl"
    made: list[ScriptedModel] = []

    def make() -> ScriptedModel:
        made.append(ScriptedModel(replies()))
        return made[-1]

    out = io.StringIO()
    code = await run_bench(
        bench_set, make=make, choice=Choice("anthropic", "claude-haiku-5-5"), sandbox=LocalSandbox(),
        config=cfg(tmp_path, fixture_venv), repeat=2, label="base", max_cost=1.0, ledger=ledger, out=out, err=io.StringIO(),
    )
    assert code == 0, out.getvalue()
    lines = bench.read_ledger(ledger)
    assert [(x["kata"], x["repeat"]) for x in lines] == [("alpha", 1), ("beta", 1), ("alpha", 2), ("beta", 2)]
    assert all(x["label"] == "base" and x["set"] == "two" and x["summary"]["bench"] == "base" for x in lines)
    assert all(x["summary"]["totals"]["green"] == 1 and x["summary"]["no_hints"] for x in lines)
    assert len(made) == 4  # a fresh model per run
    assert not any((tmp_path / "ws").iterdir())  # every Workspace removed
    for pair in bench_set.pairs:
        assert len(worktrees(pair.project)) == 1  # only the main checkout is left
        assert not subprocess.run(["git", "status", "--porcelain", "src"], cwd=pair.project, capture_output=True, text=True).stdout
    assert out.getvalue().count(" Green · ") == 4


async def test_the_bench_wide_max_cost_stops_cleanly_with_exit_3_keeping_the_runs_written(
    fixture_venv: Path, tmp_path: Path,
) -> None:
    bench_set = two_katas(tmp_path)
    ledger = tmp_path / "bench.jsonl"
    out = io.StringIO()
    code = await run_bench(
        bench_set, make=lambda: CostlyModel(replies(), price=0.06), choice=Choice("anthropic", "claude-haiku-5-5"),
        sandbox=LocalSandbox(), config=cfg(tmp_path, fixture_venv, max_cost_usd=1.0), repeat=3, label="capped",
        max_cost=0.25, ledger=ledger, out=out, err=io.StringIO(),
    )
    # run 1 costs 0.18 (three calls); run 2 gets the 0.07 left as its Spend cap and stops after its second call
    assert code == SPEND_CAP_EXIT
    lines = bench.read_ledger(ledger)
    assert [x["summary"]["status"] for x in lines] == ["kept", "spend_cap"]  # "no" at finish keeps, then removes
    assert lines[0]["summary"]["totals"]["cost_usd"] == pytest.approx(0.18)
    assert "bench stopped" in out.getvalue()
    assert not any((tmp_path / "ws").iterdir())


# --- gates 4 and 5: the comparison and old summaries -----------------------------------------------------------------

def line(label: str, kata: str, repeat: int, green: int, planned: int = 6, cost: float = 0.01, **summary: Any) -> dict[str, Any]:
    return {"label": label, "set": "s", "repeat": repeat, "kata": kata, "summary": {
        "planned": planned, "stuck": planned - green,
        "totals": {"green": green, "attempts": 4, "satisfied": 1, "seconds": 60.0, "cost_usd": cost}, **summary,
    }}


def test_the_comparison_computes_means_and_ranges_and_labels_noise() -> None:
    lines = [
        line("a", "k1", 1, 6), line("a", "k1", 2, 4), line("a", "k1", 3, 5),  # mean 5, range 4-6
        line("a", "k2", 1, 6), line("a", "k2", 2, 6), line("a", "k2", 3, 6),
        line("b", "k1", 1, 6), line("b", "k1", 2, 6), line("b", "k1", 3, 5),  # mean 5.67: within a's range of 2
        line("b", "k2", 1, 3), line("b", "k2", 2, 3), line("b", "k2", 3, 3),  # mean 3: a clear drop
    ]
    a = bench.aggregate(lines, "a")
    assert (a["k1"].green, a["k1"].green_min, a["k1"].green_max, a["k1"].n) == (5, 4, 6, 3)
    assert a["total"].green == pytest.approx(11) and (a["total"].green_min, a["total"].green_max) == (10, 12)
    assert a["k1"].cost == pytest.approx(0.01) and a["total"].cost == pytest.approx(0.02)
    b = bench.aggregate(lines, "b")
    assert bench.verdict(a["k1"], b["k1"]) == "+0.7 within noise"
    assert bench.verdict(a["k2"], b["k2"]) == "-3.0 worse"
    assert bench.verdict(a["k2"], a["k2"]) == "same"
    edge = [line("c", "k", r, g) for r, g in ((1, 4), (2, 6))] + [line("d", "k", r, 7) for r in (1, 2)]
    assert bench.verdict(bench.aggregate(edge, "c")["k"], bench.aggregate(edge, "d")["k"]) == "+2.0 better"  # not smaller than 6-4
    table = bench.compare_table(lines, "a", "b")
    assert "| k1 |" in table and "within noise" in table and "| total |" in table and "[4–6]" in table


def test_one_run_per_label_says_the_noise_is_unknown() -> None:
    lines = [line("a", "k", 1, 5), line("b", "k", 1, 6)]
    assert bench.verdict(bench.aggregate(lines, "a")["k"], bench.aggregate(lines, "b")["k"]) == "+1.0 (one run: noise unknown)"


def test_an_incomplete_repeat_is_left_out_of_the_total() -> None:
    lines = [line("a", "k1", 1, 6), line("a", "k2", 1, 6), line("a", "k1", 2, 2)]  # repeat 2 stopped after k1
    stats = bench.aggregate(lines, "a")
    assert stats["total"].n == 1 and stats["total"].green == 12 and stats["k1"].n == 2


def test_old_summaries_without_provenance_still_load(tmp_path: Path) -> None:
    totals: dict[str, Any] = {"green": 6, "attempts": 2, "red_rejections": 0, "hints": 0, "satisfied": 4, "seconds": 120.0}
    old: dict[str, Any] = {  # an M13/F1 summary: no provenance fields at all
        "model": "gemma4:e4b", "kata": "bowling", "no_hints": True, "status": "done", "planned": 6, "stuck": 0, "totals": totals,
    }
    new = {**old, "model": "claude-haiku-5-5", "plan_sha256": "abcdef0123", "totals": {**totals, "cost_usd": 0.004}}
    runs = tmp_path / "bowling" / "runs"
    runs.mkdir(parents=True)
    (runs / "20261009-000000-gemma4_e4b.json").write_text(json.dumps(old))
    (runs / "20261011-000000-claude-haiku-5-5.json").write_text(json.dumps(new))
    table = bench.summaries_table(bench.load_summaries(tmp_path))
    assert "6/6 · 2a · 0r · 0h · 4s · 0.3m |" in table  # the old one, with no cost or Plan hash
    assert "plan abcdef0" in table and "$0.004" in table
    # and a ledger line whose summary lacks provenance aggregates like any other
    assert bench.aggregate([{"label": "x", "repeat": 1, "kata": "bowling", "summary": old}], "x")["bowling"].green == 6


def test_the_manifest_resolves_plans_and_projects(tmp_path: Path) -> None:
    manifest = tmp_path / "m.toml"
    manifest.write_text('practice_dir = "/katas"\n[sets.s]\ndescription = "d"\n'
                        'pairs = [{ kata = "k", plan = "plans/k.json" }, { kata = "j", plan = "j.json", project = "/elsewhere/j" }]\n')
    [s] = bench.load_manifest(manifest).values()
    assert s.pairs[0] == bench.Pair("k", (tmp_path / "plans" / "k.json").resolve(), Path("/katas/k"))
    assert s.pairs[1].project == Path("/elsewhere/j")


def test_a_set_that_holds_a_kata_twice_needs_pair_names(tmp_path: Path) -> None:
    manifest = tmp_path / "m.toml"
    manifest.write_text('[sets.s]\npairs = [{ kata = "k", plan = "a.json" }, { kata = "k", plan = "b.json" }]\n')
    with pytest.raises(ValueError, match="give the pairs a `name`"):
        bench.load_manifest(manifest)
    manifest.write_text('[sets.s]\npairs = [{ kata = "k", name = "k/a", plan = "a.json" }, { kata = "k", name = "k/b", plan = "b.json" }]\n')
    assert [p.key for p in bench.load_manifest(manifest)["s"].pairs] == ["k/a", "k/b"]


def test_ledger_lines_aggregate_by_pair_name_when_there_is_one() -> None:
    lines = [{**line("a", "k", 1, 6), "pair": "k/a"}, {**line("a", "k", 1, 2), "pair": "k/b"}]
    stats = bench.aggregate(lines, "a")
    assert stats["k/a"].green == 6 and stats["k/b"].green == 2 and stats["total"].green == 8


def test_the_checked_in_manifest_names_the_f1_sets() -> None:
    sets = bench.load_manifest()
    assert {"claude-plans", "local-plans", "disputed", "disputed-control"} <= sets.keys()
    assert [p.key for p in sets["disputed"].pairs] == [p.key for p in sets["disputed-control"].pairs]
    assert all(Plan.load(p.plan).invariants for p in sets["disputed"].pairs)
    assert not any(Plan.load(p.plan).invariants for p in sets["disputed-control"].pairs)
    for s in sets.values():
        for pair in s.pairs:
            assert pair.plan.exists(), pair.plan
            Plan.load(pair.plan)
