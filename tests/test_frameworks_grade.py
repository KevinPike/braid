"""F13 gate: the grading adapter (a framework runner's output directory -> F7's hidden grade). No network, no model."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from harness.sandbox import LocalSandbox
import harness.workflow.bench as bench

EXOTIC = bench.HARNESS_ROOT / "bench" / "exotic"
KATA = "ferry-fare"
OWN_TEST = """from ferry_fare import fare


def test_examples() -> None:
    assert fare([30, 8], "car") == 22
    assert fare([70, 40, 10, 2], "car") == 20


def test_waiver_and_carpool() -> None:
    assert fare([30, 30, 30, 30], "none") == 21
    assert fare([30, 30, 30], "car") == 27
"""


def grade_module() -> ModuleType:
    path = bench.HARNESS_ROOT / "examples" / "frameworks" / "grade.py"
    if "frameworks_grade" in sys.modules:
        return sys.modules["frameworks_grade"]
    spec = importlib.util.spec_from_file_location("frameworks_grade", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["frameworks_grade"] = module  # its dataclasses look their module up while it loads
    writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ under examples/
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes
    return module


def hidden_total(fixture_venv: Path, tmp_path: Path) -> int:
    """What the hidden suite collects against the reference, counted here independently of the adapter."""
    scratch = tmp_path / "count"
    scratch.mkdir()
    shutil.copy(EXOTIC / KATA / "reference" / "ferry_fare.py", scratch)
    shutil.copy(EXOTIC / KATA / "hidden" / "test_hidden.py", scratch)
    done = subprocess.run([str(fixture_venv / "bin" / "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_hidden.py"],
                          cwd=scratch, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
    return bench._count(r"(\d+) passed", done.stdout)


def runner_output(root: Path, module_source: Path | None, *, own_test: bool = True) -> Path:
    """A runner's output directory as a framework leaves it: the module, a test under tests/, GOAL.md and a trace dir,
    with no pyproject.toml (the adapter supplies the kata repo's)."""
    out = root / "out"
    (out / "tests").mkdir(parents=True)
    (out / ".trace").mkdir()
    (out / ".trace" / "test_trace_is_ignored.py").write_text("def test_never_collected() -> None:\n    assert False\n")
    (out / "GOAL.md").write_text((EXOTIC / KATA / "GOAL.md").read_text())
    if module_source is not None:
        shutil.copy(module_source, out / "ferry_fare.py")
    if own_test:
        (out / "tests" / "test_ferry_fare.py").write_text(OWN_TEST)
    return out


async def grade(out: Path, fixture_venv: Path) -> dict[str, Any]:
    graded = await grade_module().grade_dir(out, KATA, sandbox=LocalSandbox(), venv=fixture_venv)
    fields: dict[str, Any] = graded.fields()
    return fields


async def test_the_reference_passes_every_hidden_test(fixture_venv: Path, tmp_path: Path) -> None:
    out = runner_output(tmp_path, EXOTIC / KATA / "reference" / "ferry_fare.py")
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    got = await grade(out, fixture_venv)
    total = hidden_total(fixture_venv, tmp_path)
    assert total > 0
    assert got["hidden_passed"] == got["hidden_total"] == total
    assert got["own_tests"] == 1 and got["mutants_total"] == 5
    assert got["mutants_killed"] >= 1  # the runner's own tests tell some mutant apart
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before  # the output is never written


async def test_a_mutant_scores_lower(fixture_venv: Path, tmp_path: Path) -> None:
    reference = await grade(runner_output(tmp_path / "ref", EXOTIC / KATA / "reference" / "ferry_fare.py"), fixture_venv)
    for mutant in sorted((EXOTIC / KATA / "mutants").glob("ferry_fare_m*.py"))[:2]:
        got = await grade(runner_output(tmp_path / mutant.stem, mutant, own_test=False), fixture_venv)
        assert got["hidden_total"] == reference["hidden_total"]
        assert got["hidden_passed"] < reference["hidden_passed"], mutant.name
        assert got["own_tests"] == 0 and got["mutants_killed"] == 0  # no tests of its own kill nothing


async def test_a_missing_module_scores_zero_of_the_whole_suite(fixture_venv: Path, tmp_path: Path) -> None:
    got = await grade(runner_output(tmp_path, None), fixture_venv)
    assert got["hidden_passed"] == 0
    assert got["hidden_total"] == hidden_total(fixture_venv, tmp_path)  # 0/N, not 0/0
    assert got["mutants_killed"] == 0


async def test_a_missing_output_directory_scores_zero(fixture_venv: Path, tmp_path: Path) -> None:
    got = await grade(tmp_path / "never-written", fixture_venv)
    assert got["hidden_passed"] == 0 and got["hidden_total"] > 0 and got["own_tests"] == 0


def test_an_unknown_kata_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an exotic kata"):
        grade_module().kata_dir("roman-numerals")
