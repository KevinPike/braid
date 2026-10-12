"""F7 gates: exotic katas graded by Hidden tests the model never sees, and its own tests graded by mutants."""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from harness.sandbox import LocalSandbox, RunResult, SandboxPolicy, Workspace
from harness.workflow import bench
from harness.workflow.backend import Choice
from harness.workflow.cli import run_bench
from harness.workflow.plan import Plan
from harness.workflow.records import read_log
from tests.conftest import git
from tests.workflow.test_api_runs import cfg
from tests.workflow.test_bench import line
from tests.workflow.test_execute import ScriptedModel, edit

EXOTIC = bench.HARNESS_ROOT / "bench" / "exotic"
CANARY = re.compile(r"braid-hidden-canary: ([a-z0-9-]+-[0-9a-f]{16})")
CANARIES = {k.name: CANARY.findall((k / "hidden" / "test_hidden.py").read_text())[0] for k in EXOTIC.iterdir() if (k / "GOAL.md").is_file()}

# What a scripted model writes for ferry-fare: its first Behavior's test, and an implementation that knows the age
# bands and vehicle fees but not the group waiver, the carpool or the errors, so some hidden tests fail.
FARE_TEST = edit("tests/test_behavior_1.py", 'from ferry_fare import fare\n\n\ndef test_b1() -> None:\n'
                 '    assert fare([50], "none") == 7\n    assert fare([2], "none") == 0\n'
                 '    assert fare([16], "none") == 7\n')  # the age-16 boundary: kills mutant 1 and no other
FARE_GRADE = bench.Grade(hidden_passed=22, hidden_total=46, mutants_killed=1, mutants_total=5)
FARE_IMPL = '''"""Price one party's crossing on the Skelling ferry."""

FEES = {"none": 0, "bike": 2, "car": 12, "van": 18}


def fare(ages: list[int], vehicle: str) -> int:
    def one(age: int) -> int:
        return 0 if age <= 3 else 3 if age <= 15 else 7 if age <= 64 else 4

    return sum(one(a) for a in ages) + FEES[vehicle]
'''


def script(path: Path) -> ModuleType:
    """A script outside the package (``make_katas.py``, ``verify.py``), imported once."""
    if path.stem in sys.modules:
        return sys.modules[path.stem]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module  # its dataclasses look their module up while it loads
    writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ under bench/ or examples/
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes
    return module


def make_katas() -> ModuleType:
    return script(bench.HARNESS_ROOT / "examples" / "graphs" / "make_katas.py")


def exotic_repo(root: Path, kata: str = "ferry-fare") -> Path:
    """The exotic kata repo ``make_katas.py`` creates, without ``uv sync`` (the tests use the fixture venv)."""
    mk = make_katas()
    summary, goal_md = mk.exotic_katas()[kata]
    mk.make(root, kata, summary, goal_md, sync=False)
    return root / kata


def files_under(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


# --- gate 1: verify.py ------------------------------------------------------------------------------------------------

def test_verify_passes_on_every_exotic_kata() -> None:
    done = subprocess.run([sys.executable, str(EXOTIC / "verify.py")], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "6/6 katas verified" in done.stdout


def test_the_exotic_set_names_six_katas_with_their_hidden_tests() -> None:
    exotic = bench.load_manifest()["exotic"]
    assert sorted(p.kata for p in exotic.pairs) == sorted(CANARIES)
    for pair in exotic.pairs:
        assert pair.hidden == (EXOTIC / pair.kata / "hidden").resolve()
        assert (pair.hidden / bench.HIDDEN_FILE).is_file() and pair.plan == (EXOTIC / pair.kata / "plan.json").resolve()
        Plan.load(pair.plan)
    assert all(p.hidden is None for name, s in bench.load_manifest().items() if name != "exotic" for p in s.pairs)


def test_make_katas_copies_only_the_goal_and_summary(tmp_path: Path) -> None:
    mk = make_katas()
    katas = mk.exotic_katas()
    assert sorted(katas) == sorted(CANARIES)
    for kata, (summary, goal_md) in katas.items():
        mk.make(tmp_path, kata, summary, goal_md, sync=False)
        repo = tmp_path / kata
        module = kata.replace("-", "_")
        tracked = git(repo, "ls-files").split()
        assert sorted(tracked) == sorted([".gitignore", "GOAL.md", "pyproject.toml", f"{module}.py", "tests/.gitkeep"])
        assert (repo / "GOAL.md").read_text() == (EXOTIC / kata / "GOAL.md").read_text()
        assert (repo / f"{module}.py").read_text() == f'"""{(EXOTIC / kata / "summary.txt").read_text().strip()}"""\n'
        for content in files_under(repo).values():
            assert not any(c.encode() in content for c in CANARIES.values())


# --- gates 2 and 4: a scripted bench run on an exotic kata ---------------------------------------------------------

@dataclass
class ExoticRun:
    code: int
    out: str
    lines: list[dict[str, Any]]
    model: ScriptedModel
    runs_dir: Path
    repo: Path


async def exotic_bench(tmp_path: Path, fixture_venv: Path) -> ExoticRun:
    repo = exotic_repo(tmp_path / "katas")
    real = Plan.load(EXOTIC / "ferry-fare" / "plan.json")
    plan_path = tmp_path / "plans" / "ferry-fare.json"
    plan_path.parent.mkdir(parents=True)
    Plan(goal=real.goal, behaviors=real.behaviors[:1]).save(plan_path)
    pair = bench.Pair("ferry-fare", plan_path, repo, hidden=EXOTIC / "ferry-fare" / "hidden")
    model = ScriptedModel([FARE_TEST, edit("ferry_fare.py", FARE_IMPL), edit("ferry_fare.py", FARE_IMPL)])
    config = cfg(tmp_path, fixture_venv)
    ledger = tmp_path / "bench.jsonl"
    out = io.StringIO()
    code = await run_bench(
        bench.BenchSet("exotic", "", (pair,)), make=lambda: model, choice=Choice("anthropic", "claude-haiku-5-5"),
        sandbox=LocalSandbox(), config=config, repeat=1, label="exo", max_cost=1.0, ledger=ledger, out=out, err=out,
    )
    return ExoticRun(code, out.getvalue(), bench.read_ledger(ledger), model, config.workflow.runs_dir, repo)


async def test_no_prompt_in_the_run_log_contains_a_hidden_canary(fixture_venv: Path, tmp_path: Path) -> None:
    run = await exotic_bench(tmp_path, fixture_venv)
    assert run.code == 0, run.out
    records = [r for log in sorted(run.runs_dir.glob("*.jsonl")) for r in read_log(log)]
    prompts = [r["prompt"] for r in records if "prompt" in r]
    assert prompts and len(prompts) == len(run.model.prompts)  # write_test, write_impl and polish each logged their prompt
    hidden_text = (EXOTIC / "ferry-fare" / "hidden" / "test_hidden.py").read_text()
    for prompt in [*prompts, *run.model.prompts]:
        assert not any(canary in prompt for canary in CANARIES.values())
        assert "test_hidden" not in prompt and hidden_text not in prompt
    for content in files_under(run.repo).values():  # nor did the hidden suite reach the kata repo
        assert CANARIES["ferry-fare"].encode() not in content
    assert run.lines[0]["summary"]["hidden_passed"] > 0  # and yet the run was graded by it


async def test_the_summary_and_ledger_carry_the_grade(fixture_venv: Path, tmp_path: Path) -> None:
    run = await exotic_bench(tmp_path, fixture_venv)
    assert run.code == 0, run.out
    [ledger_line] = run.lines
    summary = ledger_line["summary"]
    assert summary["totals"]["green"] == 1 and summary["status"] == "kept"
    # the partial implementation passes the age-band and vehicle-fee cases and fails the rest
    assert {k: summary[k] for k in bench.GRADE_FIELDS} == {k: FARE_GRADE.fields()[k] for k in bench.GRADE_FIELDS}
    assert summary["hidden_total"] == collected_count(EXOTIC / "ferry-fare")
    assert len(summary["hidden_failed"]) == FARE_GRADE.hidden_total - FARE_GRADE.hidden_passed  # which ones, by id
    # the summary file on disk says the same as the ledger
    written = bench.load_summaries(run.repo.parent)[("claude-haiku-5-5", "ferry-fare")]
    assert {k: written[k] for k in bench.GRADE_FIELDS} == {k: summary[k] for k in bench.GRADE_FIELDS}
    assert f"hidden {summary['hidden_passed']}/{summary['hidden_total']}" in run.out


def collected_count(kata: Path) -> int:
    """How many cases the hidden suite holds: what the reference passes, run the way verify.py runs it."""
    verify = script(EXOTIC / "verify.py")
    module = kata.name.replace("-", "_")
    code, passed, failed = verify.run_hidden(kata, kata / "reference" / f"{module}.py", module)
    assert (code, failed) == (0, 0)
    return int(passed)


def test_older_ledger_lines_without_the_grade_still_load() -> None:
    graded = {"hidden_passed": 30, "hidden_total": 40, "mutants_killed": 3, "mutants_total": 5}
    lines = [
        line("old", "k", 1, 6), line("old", "k", 2, 5),  # before F7: no grade at all
        line("new", "k", 1, 6, **graded),  # Green on all 6 of its own, failing 10 hidden: in the self/hidden gap
        line("new", "k", 2, 5, **{**graded, "hidden_passed": 20, "mutants_killed": 5}),  # not Green on its own: not
    ]
    old, new = bench.aggregate(lines, "old"), bench.aggregate(lines, "new")
    assert old["k"].hidden is None and (old["k"].gap, old["k"].graded) == (0, 0) and old["k"].mutants is None
    assert new["k"].hidden == pytest.approx(0.625) and new["k"].mutants == 4 and new["k"].mutants_total == 5
    assert (new["k"].gap, new["k"].graded) == (1, 2) and (new["total"].gap, new["total"].graded) == (1, 2)
    assert new["total"].hidden == pytest.approx(0.625)
    old_table = bench.label_table(lines, "old")
    assert "hidden" not in old_table.lower() and "| k | 2 |" in old_table  # the F2 table, unchanged
    new_table = bench.label_table(lines, "new")
    assert "| hidden | self/hidden gap | mutants killed |" in new_table and "| 62% | 1/2 runs | 4.0/5 |" in new_table
    compared = bench.compare_table(lines, "old", "new")
    assert "| - · - · - | 62% · 1/2 runs · 4.0/5 |" in compared


def test_a_total_needs_every_kata_graded() -> None:
    graded = {"hidden_passed": 10, "hidden_total": 20, "mutants_killed": 2, "mutants_total": 5}
    stats = bench.aggregate([line("x", "a", 1, 6, **graded), line("x", "b", 1, 6)], "x")
    assert stats["a"].hidden == 0.5 and stats["b"].hidden is None and stats["total"].hidden is None
    both = bench.aggregate([line("x", "a", 1, 6, **graded), line("x", "b", 1, 3, **graded)], "x")
    assert both["total"].hidden == 0.5 and both["total"].mutants == 4 and both["total"].mutants_total == 10
    assert (both["total"].gap, both["total"].graded) == (1, 2)  # counted kata by kata: only a was Green on its own


# --- gate 3: grading runs in the Sandbox and leaves the tests alone --------------------------------------------------

@dataclass
class RecordingSandbox:
    """Delegates to LocalSandbox; records each command, where it ran, its policy and the files it saw."""

    calls: list[tuple[str, Path, SandboxPolicy, list[str]]] = field(default_factory=list)

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = 120.0,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        seen = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file() and ".venv" not in p.relative_to(cwd).parts)
        self.calls.append((command, cwd, policy, seen))
        return await LocalSandbox().run(command, cwd=cwd, policy=policy, timeout_s=timeout_s, env=env)


async def test_grading_runs_in_the_sandbox_on_scratch_copies(fixture_venv: Path, tmp_path: Path) -> None:
    repo = exotic_repo(tmp_path / "katas")
    ws = await Workspace.create(repo, tmp_path / "ws", "graded")
    try:
        (ws.path / "ferry_fare.py").write_text(FARE_IMPL)  # the run's final code and its own test
        (ws.path / "tests" / "test_behavior_1.py").write_text(FARE_TEST.edits[0].content)
        before = (files_under(repo), files_under(ws.path))
        sandbox = RecordingSandbox()
        graded = await bench.grade(
            ws, EXOTIC / "ferry-fare" / "hidden", "ferry_fare", sandbox=sandbox,
            test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
        )
        assert (files_under(repo), files_under(ws.path)) == before  # the project's and the Workspace's tests unchanged
    finally:
        await ws.remove()
    assert replace(graded, hidden_failed=()) == FARE_GRADE
    assert len(graded.hidden_failed) == FARE_GRADE.hidden_total - FARE_GRADE.hidden_passed
    assert all("::" not in t and not t.startswith("test_hidden") for t in graded.hidden_failed)  # bare ids
    # collect (against the reference), hidden suite, the model's suite on its own code, then one run per mutant
    assert len(sandbox.calls) == 3 + 5
    for command, cwd, policy, _seen in sandbox.calls:
        assert command.startswith(f"{fixture_venv}/bin/python -m pytest")
        assert policy.writable == (cwd, cwd / ".tmp") and not policy.network
        for root in (repo.resolve(), ws.path):
            assert root not in cwd.parents and cwd != root
        assert not cwd.exists()  # each scratch copy is removed
    hidden_seen = sandbox.calls[1][3]
    assert "test_hidden.py" in hidden_seen and "tests/test_behavior_1.py" not in hidden_seen  # final non-test files only
    mutant_seen = sandbox.calls[3][3]
    assert "tests/test_behavior_1.py" in mutant_seen and "test_hidden.py" not in mutant_seen


async def test_a_suite_that_fails_on_its_own_code_kills_no_mutant(fixture_venv: Path, tmp_path: Path) -> None:
    repo = exotic_repo(tmp_path / "katas")
    ws = await Workspace.create(repo, tmp_path / "ws", "red")
    try:
        (ws.path / "ferry_fare.py").write_text(FARE_IMPL)
        (ws.path / "tests" / "test_wrong.py").write_text("from ferry_fare import fare\n\n\ndef test_w() -> None:\n    assert fare([], 'none') == 1\n")
        graded = await bench.grade(
            ws, EXOTIC / "ferry-fare" / "hidden", "ferry_fare", sandbox=LocalSandbox(),
            test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
        )
    finally:
        await ws.remove()
    assert (graded.mutants_killed, graded.mutants_total) == (0, 5)


async def test_code_that_does_not_import_scores_zero_of_the_whole_suite(fixture_venv: Path, tmp_path: Path) -> None:
    repo = exotic_repo(tmp_path / "katas")
    ws = await Workspace.create(repo, tmp_path / "ws", "empty")
    try:  # the kata's empty module: no `fare` to import
        graded = await bench.grade(
            ws, EXOTIC / "ferry-fare" / "hidden", "ferry_fare", sandbox=LocalSandbox(),
            test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
        )
    finally:
        await ws.remove()
    assert graded.hidden_passed == 0 and graded.hidden_total == collected_count(EXOTIC / "ferry-fare")
    assert graded.mutants_killed == 0  # no tests: nothing to kill a mutant with


def test_failed_ids_come_from_the_short_summary() -> None:
    output = (".F.E\n=== short test summary info ===\n"
              "FAILED test_hidden.py::test_rounds[requests3-5-expected3] - assert [1, 2] == [2, 1]\n"
              "FAILED test_hidden.py::test_elders_never_greedy\n"
              "ERROR test_hidden.py::test_errors[requests0-3] - TypeError: boom\n"
              "FAILED test_hidden.py::test_elders_never_greedy\n"
              "2 failed, 1 passed, 1 error in 0.10s\n")
    assert bench.failed_ids(output) == ("test_rounds[requests3-5-expected3]", "test_elders_never_greedy", "test_errors[requests0-3]")
    assert bench.failed_ids("ERROR test_hidden.py - ImportError: cannot import name 'fare'\n") == ("test_hidden.py",)
    assert bench.failed_ids("4 passed in 0.02s\n") == ()


def test_the_report_lists_which_hidden_tests_failed_and_how_often() -> None:
    graded = {"hidden_passed": 3, "hidden_total": 5, "mutants_killed": 2, "mutants_total": 5}
    lines = [
        line("x", "k", 1, 6, **graded, hidden_failed=["test_a[1]", "test_b"]),
        line("x", "k", 2, 6, **graded, hidden_failed=["test_a[1]"]),
        line("x", "k", 3, 6, **graded),  # an older line, graded before the ids were recorded: not counted
        line("x", "j", 1, 6, **{**graded, "hidden_passed": 5}, hidden_failed=[]),
    ]
    table = bench.label_table(lines, "x")
    assert "Hidden tests failed (runs failing each / runs recorded):\n- k: test_a[1] (2/2), test_b (1/2)" in table
    assert "- j:" not in table
    assert bench.hidden_failures([line("x", "k", 1, 6, **graded)], "x") == ""  # nothing recorded: no section


def test_test_files_are_told_apart_from_code() -> None:
    assert all(bench.is_test_file(p) for p in ("tests/test_a.py", "tests/helpers.py", "test_x.py", "pkg/x_test.py", "conftest.py"))
    assert not any(bench.is_test_file(p) for p in ("ferry_fare.py", "pyproject.toml", "GOAL.md", "pkg/testing.py"))
