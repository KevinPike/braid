"""F2 Benchmarks: know whether a change helped.

A manifest (``bench/katas.toml``) names sets of (kata, Plan) pairs; ``harness-workflow bench`` runs a set N times and
appends one line per run to a ledger (``~/.braid/bench.jsonl``) holding the run's summary and its provenance: which
harness commit, Plan, backend and settings produced it. ``harness-workflow report`` reads the ledger back: per kata and
in total, the mean and min-max range over repeats, and whether a difference between two labels is bigger than that
range or "within noise". Comparison only: nothing here is a gate, and a drop never fails the test suite.

F7 adds Hidden tests: a pair may name a held-out suite (``bench/exotic/<kata>/hidden``), and ``grade`` scores the final
Workspace against it, and the model-written tests against each mutant, in scratch copies run in the Sandbox.

F12 adds the frontier: ``report --frontier`` sets each label of a model x effort sweep against the others, hidden pass
rate against cost per kata.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any

from harness.sandbox import RunResult, Sandbox, SandboxPolicy, Workspace, toolchain_paths
from harness.workflow.plan import Plan

HARNESS_ROOT = Path(__file__).resolve().parents[3]  # src/harness/workflow/bench.py -> the repo
DEFAULT_MANIFEST = HARNESS_ROOT / "bench" / "katas.toml"


# --- provenance ----------------------------------------------------------------------------------------------------

def harness_commit(root: Path = HARNESS_ROOT) -> tuple[str | None, bool | None]:
    """(HEAD commit, whether the tree has uncommitted changes); (None, None) outside a git checkout."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(dirty.strip())


def plan_sha256(plan: Plan) -> str:
    """A hash of the Plan's content (canonical JSON), so reformatting a file does not change it but any edit does."""
    return hashlib.sha256(json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def provenance(
    plan: Plan, *, plan_path: Path | None, backend: str, effort: str | None, max_attempts: int, execution: str,
    bench: str | None, root: Path = HARNESS_ROOT,
) -> dict[str, Any]:
    """The summary fields that say what produced a run. Plain data: summaries from before F2 load without them."""
    commit, dirty = harness_commit(root)
    return {
        "harness_commit": commit, "harness_dirty": dirty,
        "plan_path": str(plan_path) if plan_path else None, "plan_sha256": plan_sha256(plan),
        "backend": backend, "effort": effort, "max_attempts": max_attempts, "execution": execution, "bench": bench,
    }


# --- the manifest ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    kata: str
    plan: Path
    project: Path
    name: str = ""  # how the ledger and the report name it; the kata unless a set holds the kata twice
    hidden: Path | None = None  # F7: the kata's Hidden tests (``<kata>/hidden``, with its mutants and reference beside it)

    @property
    def key(self) -> str:
        return self.name or self.kata

    @property
    def module(self) -> str:
        """The module under test, as ``make_katas.py`` names it."""
        return self.kata.replace("-", "_")


@dataclass(frozen=True)
class BenchSet:
    name: str
    description: str
    pairs: tuple[Pair, ...]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, BenchSet]:
    """``[sets.<name>]`` tables of ``pairs = [{kata, plan, project?, name?, hidden?}]``. Plan and hidden paths are
    relative to the manifest; a kata's project defaults to ``practice_dir/<kata>``; ``name`` (default the kata) must be
    unique within a set."""
    data = tomllib.loads(path.read_text())
    practice = Path(data.get("practice_dir", "~/projects/braid-practice")).expanduser()
    sets: dict[str, BenchSet] = {}
    for name, body in data.get("sets", {}).items():
        pairs = tuple(
            Pair(p["kata"], (path.parent / p["plan"]).resolve(),
                 Path(p["project"]).expanduser() if "project" in p else practice / p["kata"], p.get("name", ""),
                 (path.parent / p["hidden"]).resolve() if "hidden" in p else None)
            for p in body["pairs"]
        )
        keys = [x.key for x in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError(f"set {name!r} names a pair twice ({', '.join(keys)}); give the pairs a `name`")
        sets[name] = BenchSet(name, body.get("description", ""), pairs)
    return sets


# --- the ledger ------------------------------------------------------------------------------------------------------

def ledger_line(*, label: str, set_name: str, repeat: int, pair: Pair, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label, "set": set_name, "repeat": repeat, "kata": pair.kata, "pair": pair.key, "plan": str(pair.plan),
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "summary": dict(summary),
    }


def append_ledger(path: Path, line: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(line) + "\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


# --- Hidden tests (F7) -----------------------------------------------------------------------------------------------

HIDDEN_FILE = "test_hidden.py"
GRADE_FIELDS = ("hidden_passed", "hidden_total", "mutants_killed", "mutants_total")


@dataclass(frozen=True)
class Grade:
    """A run's final code against the Hidden tests, and its own tests against the mutants."""

    hidden_passed: int
    hidden_total: int  # what the hidden suite collects against the reference, so code that cannot import scores 0/N
    mutants_killed: int  # mutants the model-written tests fail (pytest exit 1), counted only if they pass on the run's code
    mutants_total: int

    def fields(self) -> dict[str, int]:
        return asdict(self)


def is_test_file(path: str) -> bool:
    """A file the model wrote as a test (or test setup): left out when the final code meets the Hidden tests."""
    p = PurePosixPath(path)
    return (p.parts[0] in ("tests", "test") or p.name == "conftest.py"
            or (p.suffix == ".py" and (p.name.startswith("test_") or p.stem.endswith("_test"))))


def _count(pattern: str, output: str) -> int:
    found = re.findall(pattern, output)
    return int(found[-1]) if found else 0


async def run_in_scratch(
    files: Mapping[str, Path], command: str, *, workspace: Workspace, sandbox: Sandbox, timeout_s: float,
) -> RunResult:
    """Copy ``files`` (scratch path -> source) into a fresh scratch directory outside the Workspace and the project, and
    run ``command`` there in the Sandbox, which may write only in the scratch copy. The Workspace's ``.venv`` is linked
    in, so the run's own test command works unchanged. The scratch copy is removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="braid-grade-") as tmp:
        scratch = Path(tmp).resolve()
        for rel, source in files.items():
            (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, scratch / rel)
        venv = workspace.path / ".venv"
        if venv.exists():
            (scratch / ".venv").symlink_to(venv)
        policy = SandboxPolicy(
            writable=(scratch, scratch / ".tmp"), readable=toolchain_paths(venv) if venv.exists() else (), network=False,
        )
        env = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin", "VIRTUAL_ENV": str(venv), "PYTHONDONTWRITEBYTECODE": "1"}
        return await sandbox.run(command, cwd=scratch, policy=policy, timeout_s=timeout_s, env=env)


async def grade(
    workspace: Workspace, hidden: Path, module: str, *, sandbox: Sandbox, test_command: str, timeout_s: float = 120.0,
) -> Grade:
    """Grade the final Workspace, after ``finish`` and before it is removed. Grading is code, not a model.

    - Hidden tests: the final non-test files plus ``hidden/test_hidden.py``; ``hidden_passed`` is what pytest passes.
    - Test strength: every final file, with ``<module>.py`` replaced by each ``mutants/<module>_m*.py``. A mutant is
      killed when the model-written tests fail on it (exit 1) and pass on the run's own code.

    Nothing from ``hidden`` or beside it is copied into the Workspace or the project, only into scratch copies."""
    kata = hidden.parent
    finals = {rel: workspace.path / rel for rel in await workspace.list_files()}
    code = {rel: source for rel, source in finals.items() if not is_test_file(rel)}
    suite = hidden / HIDDEN_FILE

    async def run(files: Mapping[str, Path], target: str = "") -> RunResult:
        command = f"{test_command} {target}".strip()
        return await run_in_scratch(files, command, workspace=workspace, sandbox=sandbox, timeout_s=timeout_s)

    reference = kata / "reference" / f"{module}.py"
    collected = await run({**code, f"{module}.py": reference, HIDDEN_FILE: suite}, f"--collect-only {HIDDEN_FILE}")
    graded = await run({**code, HIDDEN_FILE: suite}, HIDDEN_FILE)
    mutants = sorted((kata / "mutants").glob(f"{module}_m*.py"))
    killed = 0
    if (await run(finals)).exit_code == 0:  # a suite that fails on the run's own code tells no mutant apart
        for mutant in mutants:
            result = await run({**finals, f"{module}.py": mutant})
            killed += result.exit_code == 1  # as in verify.py: a failing test, not a collection error or a timeout
    return Grade(_count(r"(\d+) passed", graded.output), _count(r"(\d+) tests? collected", collected.output),
                 killed, len(mutants))


# --- comparison -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Stats:
    """One kata (or the total) under one label, over its repeats."""

    n: int
    planned: int
    green: float  # mean Behaviors Green
    green_min: int
    green_max: int
    attempts: float
    satisfied: float
    stuck: float
    minutes: float
    cost: float
    # F7, over the runs graded by Hidden tests (None when none were: older lines, or a pair without them)
    hidden: float | None = None  # mean hidden pass rate, 0 to 1
    gap: int = 0  # the self/hidden gap: graded runs Green on every Behavior that still fail a hidden test
    graded: int = 0  # runs graded by Hidden tests
    mutants: float | None = None  # mean mutants killed
    mutants_total: int = 0

    @property
    def spread(self) -> int:
        return self.green_max - self.green_min


def _gapped(s: Mapping[str, Any]) -> int:
    """1 when a graded run went Green on every Behavior and still fails a hidden test (a merged total carries its count)."""
    if "gapped" in s:
        return int(s["gapped"])
    return int(s["totals"]["green"] >= s.get("planned", 0) and s["hidden_passed"] < s["hidden_total"])


def _stats(summaries: Sequence[Mapping[str, Any]]) -> Stats:
    greens = [s["totals"]["green"] for s in summaries]
    graded = [s for s in summaries if s.get("hidden_total")]
    rates = [s["hidden_passed"] / s["hidden_total"] for s in graded]
    killed = [s for s in summaries if "mutants_killed" in s]
    return Stats(
        n=len(summaries), planned=max(s.get("planned", 0) for s in summaries), green=mean(greens),
        green_min=min(greens), green_max=max(greens),
        attempts=mean(s["totals"].get("attempts", 0) for s in summaries),
        satisfied=mean(s["totals"].get("satisfied", 0) for s in summaries),
        stuck=mean(s.get("stuck", 0) for s in summaries),
        minutes=mean(s["totals"].get("seconds", 0.0) / 60 for s in summaries),
        cost=mean(s["totals"].get("cost_usd", 0.0) or 0.0 for s in summaries),
        hidden=mean(rates) if graded else None,
        gap=sum(_gapped(s) for s in graded), graded=sum(s.get("graded", 1) for s in graded),
        mutants=mean(s["mutants_killed"] for s in killed) if killed else None,
        mutants_total=max((s.get("mutants_total", 0) for s in killed), default=0),
    )


def _total(summaries_by_repeat: Mapping[int, Sequence[Mapping[str, Any]]], katas: int) -> Stats | None:
    """The total over katas, per repeat that has every kata; its range is over those repeats."""
    complete = [runs for runs in summaries_by_repeat.values() if len({r["kata"] for r in runs}) == katas]
    if not complete:
        return None
    merged = [{
        "kata": "total", "planned": sum(r.get("planned", 0) for r in runs), "stuck": sum(r.get("stuck", 0) for r in runs),
        "totals": {k: sum(r["totals"].get(k, 0) or 0 for r in runs) for k in ("green", "attempts", "satisfied", "seconds", "cost_usd")},
        **_merged_grade(runs),
    } for runs in complete]
    return _stats(merged)


def _merged_grade(runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """F7's fields for one repeat's total, when every kata in it was graded; its gap counts the runs, kata by kata."""
    if not all(r.get("hidden_total") and "mutants_killed" in r for r in runs):
        return {}
    return {**{k: sum(r[k] for r in runs) for k in GRADE_FIELDS}, "gapped": sum(_gapped(r) for r in runs), "graded": len(runs)}


def aggregate(lines: Iterable[Mapping[str, Any]], label: str) -> dict[str, Stats]:
    """Per pair (its name, else its kata), and ``"total"``, for every ledger line with ``label``."""
    by_kata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        if line.get("label") != label or line["summary"].get("status") == API_ERROR:
            continue
        key = line.get("pair", line["kata"])
        summary = {**line["summary"], "kata": key}
        by_kata[key].append(summary)
        by_repeat[line["repeat"]].append(summary)
    out = {kata: _stats(runs) for kata, runs in sorted(by_kata.items())}
    total = _total(by_repeat, len(by_kata))
    if total is not None:
        out["total"] = total
    return out


API_ERROR = "api_error"  # F8: a run an API failure stopped says nothing about the model, so it is left out of the means


def api_errors(lines: Iterable[Mapping[str, Any]], label: str) -> str:
    """The note under a table: how many of ``label``'s runs ended with ``api_error``, or nothing."""
    n = sum(1 for x in lines if x.get("label") == label and x["summary"].get("status") == API_ERROR)
    return f"{label}: {n} run(s) ended with api_error, left out of the means" if n else ""


def verdict(a: Stats, b: Stats) -> str:
    """How B's mean Green compares with A's: a difference smaller than the wider of the two ranges is noise."""
    diff = b.green - a.green
    if diff == 0:
        return "same"
    if a.n < 2 or b.n < 2:
        return f"{diff:+.1f} (one run: noise unknown)"
    if abs(diff) < max(a.spread, b.spread):
        return f"{diff:+.1f} within noise"
    return f"{diff:+.1f} {'better' if diff > 0 else 'worse'}"


def _cell(s: Stats) -> str:
    rng = f" [{s.green_min}–{s.green_max}]" if s.n > 1 else ""
    return (f"{s.green:.1f}/{s.planned}{rng} · {s.attempts:.1f}a · {s.satisfied:.1f}s · {s.stuck:.1f} stuck · "
            f"{s.minutes:.1f}m · ${s.cost:.4f}")


LEGEND = ("Cell: mean Behaviors Green/planned [min–max over repeats] · implementation attempts (a) · Satisfied (s) · "
          "Stuck · minutes per run (m) · cost per run. A difference smaller than the wider range is within noise.")
HIDDEN_LEGEND = ("Hidden tests (F7): mean hidden pass rate · self/hidden gap (graded runs Green on every Behavior that "
                 "still fail a hidden test) · mean mutants killed by the model-written tests. '-' where nothing was graded.")


def _hidden(s: Stats | None) -> tuple[str, str, str]:
    """The three F7 columns: hidden pass rate, self/hidden gap, mutants killed."""
    if s is None:
        return "-", "-", "-"
    mutants = f"{s.mutants:.1f}/{s.mutants_total}" if s.mutants is not None else "-"
    if s.hidden is None:
        return "-", "-", mutants
    return f"{s.hidden:.0%}", f"{s.gap}/{s.graded} runs", mutants


def _graded(*stats: Mapping[str, Stats]) -> bool:
    return any(s.hidden is not None or s.mutants is not None for group in stats for s in group.values())


def label_table(lines: Sequence[Mapping[str, Any]], label: str) -> str:
    stats = aggregate(lines, label)
    if not stats:
        return f"no ledger lines labelled {label!r}"
    n = max(s.n for s in stats.values())
    notes = [note for note in (api_errors(lines, label),) if note]  # F8
    if not _graded(stats):
        rows = [f"label {label}: {n} repeat(s)", "", "| kata | runs | result |", "|---|---|---|"]
        rows += [f"| {kata} | {s.n} | {_cell(s)} |" for kata, s in stats.items()]
        return "\n".join([*rows, *notes, "", LEGEND])
    rows = [f"label {label}: {n} repeat(s)", "", "| kata | runs | result | hidden | self/hidden gap | mutants killed |",
            "|---|---|---|---|---|---|"]
    rows += [f"| {kata} | {s.n} | {_cell(s)} | {' | '.join(_hidden(s))} |" for kata, s in stats.items()]
    return "\n".join([*rows, *notes, "", LEGEND, HIDDEN_LEGEND])


def compare_table(lines: Sequence[Mapping[str, Any]], label_a: str, label_b: str) -> str:
    a, b = aggregate(lines, label_a), aggregate(lines, label_b)
    missing = [x for x, s in ((label_a, a), (label_b, b)) if not s]
    if missing:
        return f"no ledger lines labelled {', '.join(map(repr, missing))}"
    graded = _graded(a, b)
    rows = [f"| kata | {label_a} | {label_b} | Green difference |", "|---|---|---|---|"]
    if graded:  # F7: hidden pass rate, self/hidden gap and mutants killed, per label
        rows = [rows[0] + f" {label_a} hidden · gap · mutants | {label_b} hidden · gap · mutants |", "|---|---|---|---|---|---|"]
    for kata in [*sorted((set(a) | set(b)) - {"total"}), "total"]:
        sa, sb = a.get(kata), b.get(kata)
        if sa is None or sb is None:
            row = f"| {kata} | {_cell(sa) if sa else '-'} | {_cell(sb) if sb else '-'} | - |"
        else:
            row = f"| {kata} | {_cell(sa)} | {_cell(sb)} | {verdict(sa, sb)} |"
        rows.append(row + (f" {' · '.join(_hidden(sa))} | {' · '.join(_hidden(sb))} |" if graded else ""))
    rows += [note for note in (api_errors(lines, label_a), api_errors(lines, label_b)) if note]  # F8
    return "\n".join([*rows, "", LEGEND, *([HIDDEN_LEGEND] if graded else [])])


@dataclass(frozen=True)
class FrontierPoint:
    """One label of a sweep (F12): what it scored on the Hidden tests and what it cost, over its katas."""

    label: str
    model: str
    effort: str
    runs: int
    hidden: float | None  # mean over katas of the mean hidden pass rate; None when nothing was graded
    cost: float  # mean cost per kata run


def frontier_points(lines: Sequence[Mapping[str, Any]], labels_: Sequence[str]) -> list[FrontierPoint]:
    """A point per label that has ledger lines, its model and effort read from its summaries' provenance."""
    points = []
    for label in labels_:
        stats = {k: s for k, s in aggregate(lines, label).items() if k != "total"}
        if not stats:
            continue
        first = next(x["summary"] for x in lines if x.get("label") == label)
        graded = [s.hidden for s in stats.values() if s.hidden is not None]
        points.append(FrontierPoint(
            label, str(first.get("model", "?")), str(first.get("effort") or "default"), sum(s.n for s in stats.values()),
            mean(graded) if graded else None, mean(s.cost for s in stats.values()),
        ))
    return points


def on_frontier(p: FrontierPoint, points: Sequence[FrontierPoint]) -> bool:
    """No other graded label passes as many hidden tests or more for as little or less, and beats it on one of them."""
    if p.hidden is None:
        return False
    return not any(
        q is not p and q.hidden is not None and q.hidden >= p.hidden and q.cost <= p.cost
        and (q.hidden > p.hidden or q.cost < p.cost)
        for q in points
    )


FRONTIER_LEGEND = ("Frontier (F12): hidden pass rate (mean over katas) against mean cost per kata run, cheapest first. "
                   "* marks a label no other label beats on both. Per kata: hidden pass rate · cost per run.")


def frontier_table(lines: Sequence[Mapping[str, Any]], labels_: Sequence[str]) -> str:
    points = frontier_points(lines, labels_)
    missing = [x for x in labels_ if x not in {p.label for p in points}]
    if not points:
        return f"no ledger lines labelled {', '.join(map(repr, missing))}"
    rows = ["| label | model | effort | runs | hidden | cost per kata | frontier |", "|---|---|---|---|---|---|---|"]
    for p in sorted(points, key=lambda p: p.cost):
        hidden = f"{p.hidden:.0%}" if p.hidden is not None else "-"
        rows.append(f"| {p.label} | {p.model} | {p.effort} | {p.runs} | {hidden} | ${p.cost:.4f} | "
                    f"{'*' if on_frontier(p, points) else ''} |")
    by_label = {p.label: {k: s for k, s in aggregate(lines, p.label).items() if k != "total"} for p in points}
    katas = sorted({k for stats in by_label.values() for k in stats})
    rows += ["", "| kata | " + " | ".join(p.label for p in points) + " |", "|---|" + "---|" * len(points)]
    for kata in katas:
        cells = []
        for p in points:
            s = by_label[p.label].get(kata)
            cells.append("-" if s is None else f"{f'{s.hidden:.0%}' if s.hidden is not None else '-'} · ${s.cost:.4f}")
        rows.append(f"| {kata} | " + " | ".join(cells) + " |")
    notes = [n for n in (api_errors(lines, p.label) for p in points) if n]  # F8
    if missing:
        notes.append(f"no ledger lines labelled {', '.join(map(repr, missing))}")
    return "\n".join([*rows, *notes, "", FRONTIER_LEGEND])


def labels(lines: Iterable[Mapping[str, Any]]) -> list[tuple[str, int, str]]:
    """(label, runs, first timestamp) per label, oldest first."""
    seen: dict[str, list[str]] = {}
    for line in lines:
        seen.setdefault(line["label"], []).append(line.get("at", ""))
    return sorted(((k, len(v), min(v)) for k, v in seen.items()), key=lambda t: t[2])


# --- the per-run summaries (the M13/F1 tables) -------------------------------------------------------------------------

def load_summaries(practice: Path, include_hinted: bool = False) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest summary per (model, kata) under ``practice/*/runs/*.json``; older summaries, without provenance, load."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(practice.glob("*/runs/*.json")):  # names start with a timestamp, so later wins
        run = json.loads(path.read_text())
        if run.get("no_hints") or include_hinted:
            latest[(run["model"], run["kata"])] = run
    return latest


def summary_cell(run: Mapping[str, Any] | None) -> str:
    if run is None:
        return "-"
    t = run["totals"]
    minutes = t["seconds"] / 60 / max(run["planned"], 1)
    cost = f" · ${t['cost_usd']:.3f}" if t.get("cost_usd") else ""  # API-backed runs only
    plan = f" · plan {run['plan_sha256'][:7]}" if run.get("plan_sha256") else ""  # F2 provenance, when present
    return (f"{t['green']}/{run['planned']} · {t['attempts']}a · {t['red_rejections']}r · {t['hints']}h · "
            f"{t.get('satisfied', 0)}s · {minutes:.1f}m{cost}{plan}")


def summaries_table(runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> str:
    models = sorted({m for m, _ in runs})
    katas = sorted({k for _, k in runs})
    lines = ["| model | " + " | ".join(katas) + " | total Green |", "|---|" + "---|" * (len(katas) + 1)]
    for model in models:
        got = [runs.get((model, k)) for k in katas]
        green = sum(r["totals"]["green"] for r in got if r)
        planned = sum(r["planned"] for r in got if r)
        lines.append(f"| {model} | " + " | ".join(summary_cell(r) for r in got) + f" | {green}/{planned} |")
    lines.append("")
    lines.append("Cell: Behaviors Green/planned · impl attempts (a) · Red rejections (r) · hints (h) · Behaviors already "
                 "satisfied by earlier code (s, counted as Green) · minutes per Behavior (m) · cost in USD ($, API-backed "
                 "runs only) · Plan hash (runs since F2).")
    return "\n".join(lines)
