"""Check every exotic kata: hidden tests, mutants, Plan Invariants, plan.json and canaries.

    uv run python bench/exotic/verify.py

For each kata under bench/exotic/ it copies the module under test and hidden/test_hidden.py into a fresh temporary
directory and runs ``sys.executable -m pytest`` there, so nothing is written under bench/. It checks that:

- the reference passes every hidden test, and the suite has 30 to 60 cases;
- each mutant fails at least one hidden test (a collection error does not count);
- every Plan Invariant holds on every (args, expected) pair the F3 extractor finds in the hidden tests;
- plan.json loads as a Plan, and no Behavior quotes a hidden call beyond the two GOAL.md examples;
- GOAL.md has exactly two worked examples;
- the hidden file starts with a ``braid-hidden-canary`` line unique to it, found in no other file under bench/exotic/.

It prints one row per kata and exits 1 if any check fails.
"""

from __future__ import annotations

import builtins
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from harness.workflow.invariants import Example, extract
from harness.workflow.plan import Plan

ROOT = Path(__file__).resolve().parent
CANARY = re.compile(r"^# braid-hidden-canary: ([a-z0-9-]+-[0-9a-f]{16})$", re.MULTILINE)
SAFE_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map", "max",
    "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "TypeError",
)  # the same builtins the Sandbox checker (harness.workflow.invariants.CHECKER) allows
SAFE = {name: getattr(builtins, name) for name in SAFE_NAMES}
MIN_CASES, MAX_CASES = 30, 60


@dataclass
class Row:
    kata: str
    cases: int = 0
    mutants: str = ""
    examples: int = 0
    invariant_checks: int = 0
    problems: list[str] = field(default_factory=list)


def run_hidden(kata_dir: Path, module_file: Path, module: str) -> tuple[int, int, int]:
    """Run the hidden suite against ``module_file`` (installed as ``module``); returns (exit code, passed, failed)."""
    with tempfile.TemporaryDirectory(prefix="braid-exotic-") as tmp:
        work = Path(tmp)
        shutil.copyfile(module_file, work / f"{module}.py")
        shutil.copyfile(kata_dir / "hidden" / "test_hidden.py", work / "test_hidden.py")
        (work / "pytest.ini").write_text("[pytest]\n")
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_hidden.py"],
            cwd=work, capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    return proc.returncode, passed, failed


def call_forms(e: Example) -> set[str]:
    """How a hidden call might be written in a Behavior: ``fn(args)`` with Python or JSON-style quotes."""
    call = f"{e.fn}({', '.join(repr(a) for a in e.args)})"
    return {call, call.replace("'", '"')}


def check_invariants(row: Row, plan: Plan, kata_dir: Path) -> None:
    hidden = kata_dir / "hidden" / "test_hidden.py"
    extraction = extract(hidden.read_text(), str(hidden.relative_to(ROOT)))
    row.examples = len(extraction.examples)
    if not extraction.examples:
        row.problems.append("the extractor found no examples in the hidden tests")
    for example in extraction.examples:
        for inv in plan.invariants:
            if inv.fn and inv.fn != example.fn:
                continue
            scope = {"__builtins__": SAFE, "fn": example.fn, "args": example.args, "kwargs": example.kwargs,
                     "result": example.expected}
            try:
                ok = bool(eval(inv.check, scope))  # the Plan's own checks, with the Sandbox checker's builtins
            except Exception as exc:  # a check that cannot evaluate is a failure, as in F3
                row.problems.append(f"invariant {inv.text!r} raised {type(exc).__name__} on {example.test}")
                continue
            row.invariant_checks += 1
            if not ok:
                row.problems.append(f"invariant {inv.text!r} is false on {example.test}")
    goal = (kata_dir / "GOAL.md").read_text()
    behaviors = "\n".join(plan.behaviors)
    for example in extraction.examples:
        forms = call_forms(example)
        if any(f in behaviors for f in forms) and not any(f in goal for f in forms):
            row.problems.append(f"a Behavior quotes the hidden call {sorted(forms)[0]}")


def check_canary(row: Row, kata_dir: Path, all_files: list[Path]) -> None:
    hidden_text = (kata_dir / "hidden" / "test_hidden.py").read_text()
    found = CANARY.findall(hidden_text)
    if len(found) != 1 or not hidden_text.startswith("# braid-hidden-canary: "):
        row.problems.append("hidden/test_hidden.py must start with exactly one canary line")
        return
    canary = found[0]
    if not canary.startswith(f"{kata_dir.name}-"):
        row.problems.append(f"canary {canary} does not name the kata")
    hidden = kata_dir / "hidden" / "test_hidden.py"
    for path in all_files:
        if path != hidden and canary in path.read_text(errors="replace"):
            row.problems.append(f"canary leaked into {path.relative_to(ROOT)}")


def verify(kata_dir: Path, all_files: list[Path]) -> Row:
    kata = kata_dir.name
    module = kata.replace("-", "_")
    row = Row(kata)
    for needed in ("GOAL.md", "summary.txt", f"reference/{module}.py", "hidden/test_hidden.py", "plan.json"):
        if not (kata_dir / needed).is_file():
            row.problems.append(f"missing {needed}")
    if row.problems:
        return row

    goal_examples = [line for line in (kata_dir / "GOAL.md").read_text().splitlines() if line.startswith("- `") and "` returns `" in line]
    if len(goal_examples) != 2:
        row.problems.append(f"GOAL.md has {len(goal_examples)} worked examples, not 2")
    if len((kata_dir / "summary.txt").read_text().strip().splitlines()) != 1:
        row.problems.append("summary.txt is not one line")

    code, passed, failed = run_hidden(kata_dir, kata_dir / "reference" / f"{module}.py", module)
    row.cases = passed + failed
    if code != 0 or failed:
        row.problems.append(f"reference: {failed} hidden tests fail (pytest exit {code})")
    if not MIN_CASES <= row.cases <= MAX_CASES:
        row.problems.append(f"{row.cases} hidden cases, outside {MIN_CASES}-{MAX_CASES}")

    mutants = sorted((kata_dir / "mutants").glob(f"{module}_m*.py"))
    killed = []
    for mutant in mutants:
        code, _, failed = run_hidden(kata_dir, mutant, module)
        if code == 1 and failed >= 1:
            killed.append(failed)
        else:
            row.problems.append(f"{mutant.name} survives (pytest exit {code}, {failed} failed)")
    row.mutants = f"{len(killed)}/{len(mutants)} ({','.join(map(str, killed))})"
    if len(mutants) != 5:
        row.problems.append(f"{len(mutants)} mutants, not 5")

    try:
        plan = Plan.load(kata_dir / "plan.json")
    except Exception as exc:
        row.problems.append(f"plan.json is not a valid Plan: {exc}")
    else:
        if not 4 <= len(plan.behaviors) <= 6:
            row.problems.append(f"{len(plan.behaviors)} Behaviors, not 4-6")
        if not 1 <= len(plan.invariants) <= 3:
            row.problems.append(f"{len(plan.invariants)} Invariants, not 1-3")
        check_invariants(row, plan, kata_dir)

    check_canary(row, kata_dir, all_files)
    return row


def main() -> int:
    katas = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "GOAL.md").is_file())
    all_files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda k: verify(k, all_files), katas))
    canaries: dict[str, str] = {}
    for kata_dir, row in zip(katas, rows, strict=True):
        for canary in CANARY.findall((kata_dir / "hidden" / "test_hidden.py").read_text()):
            if canary in canaries:
                row.problems.append(f"canary shared with {canaries[canary]}")
            canaries[canary] = kata_dir.name

    header = f"{'kata':<16} {'hidden':>6}  {'mutants killed (fails each)':<28} {'examples':>8} {'inv checks':>10}  result"
    print(header)
    print("-" * len(header))
    for row in rows:
        verdict = "ok" if not row.problems else "FAIL"
        print(f"{row.kata:<16} {row.cases:>6}  {row.mutants:<28} {row.examples:>8} {row.invariant_checks:>10}  {verdict}")
        for problem in row.problems:
            print(f"    - {problem}")
    if not katas:
        print("no katas found")
    failed = [r for r in rows if r.problems]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} katas verified")
    return 1 if failed or not katas else 0


if __name__ == "__main__":
    sys.exit(main())
