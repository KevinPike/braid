"""F2 Benchmarks: know whether a change helped.

A manifest (``bench/katas.toml``) names sets of (kata, Plan) pairs; ``harness-workflow bench`` runs a set N times and
appends one line per run to a ledger (``~/.braid/bench.jsonl``) holding the run's summary and its provenance: which
harness commit, Plan, backend and settings produced it. ``harness-workflow report`` reads the ledger back: per kata and
in total, the mean and min-max range over repeats, and whether a difference between two labels is bigger than that
range or "within noise". Comparison only: nothing here is a gate, and a drop never fails the test suite.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

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


@dataclass(frozen=True)
class BenchSet:
    name: str
    description: str
    pairs: tuple[Pair, ...]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, BenchSet]:
    """``[sets.<name>]`` tables of ``pairs = [{kata, plan, project?}]``. A Plan path is relative to the manifest; a kata's
    project defaults to ``practice_dir/<kata>``."""
    data = tomllib.loads(path.read_text())
    practice = Path(data.get("practice_dir", "~/projects/braid-practice")).expanduser()
    sets: dict[str, BenchSet] = {}
    for name, body in data.get("sets", {}).items():
        pairs = tuple(
            Pair(p["kata"], (path.parent / p["plan"]).resolve(), Path(p["project"]).expanduser() if "project" in p else practice / p["kata"])
            for p in body["pairs"]
        )
        sets[name] = BenchSet(name, body.get("description", ""), pairs)
    return sets


# --- the ledger ------------------------------------------------------------------------------------------------------

def ledger_line(*, label: str, set_name: str, repeat: int, pair: Pair, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label, "set": set_name, "repeat": repeat, "kata": pair.kata, "plan": str(pair.plan),
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "summary": dict(summary),
    }


def append_ledger(path: Path, line: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(line) + "\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


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

    @property
    def spread(self) -> int:
        return self.green_max - self.green_min


def _stats(summaries: Sequence[Mapping[str, Any]]) -> Stats:
    greens = [s["totals"]["green"] for s in summaries]
    return Stats(
        n=len(summaries), planned=max(s.get("planned", 0) for s in summaries), green=mean(greens),
        green_min=min(greens), green_max=max(greens),
        attempts=mean(s["totals"].get("attempts", 0) for s in summaries),
        satisfied=mean(s["totals"].get("satisfied", 0) for s in summaries),
        stuck=mean(s.get("stuck", 0) for s in summaries),
        minutes=mean(s["totals"].get("seconds", 0.0) / 60 for s in summaries),
        cost=mean(s["totals"].get("cost_usd", 0.0) or 0.0 for s in summaries),
    )


def _total(summaries_by_repeat: Mapping[int, Sequence[Mapping[str, Any]]], katas: int) -> Stats | None:
    """The total over katas, per repeat that has every kata; its range is over those repeats."""
    complete = [runs for runs in summaries_by_repeat.values() if len({r["kata"] for r in runs}) == katas]
    if not complete:
        return None
    merged = [{
        "kata": "total", "planned": sum(r.get("planned", 0) for r in runs), "stuck": sum(r.get("stuck", 0) for r in runs),
        "totals": {k: sum(r["totals"].get(k, 0) or 0 for r in runs) for k in ("green", "attempts", "satisfied", "seconds", "cost_usd")},
    } for runs in complete]
    return _stats(merged)


def aggregate(lines: Iterable[Mapping[str, Any]], label: str) -> dict[str, Stats]:
    """Per kata, and ``"total"``, for every ledger line with ``label``."""
    by_kata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        if line.get("label") != label:
            continue
        summary = {**line["summary"], "kata": line["kata"]}
        by_kata[line["kata"]].append(summary)
        by_repeat[line["repeat"]].append(summary)
    out = {kata: _stats(runs) for kata, runs in sorted(by_kata.items())}
    total = _total(by_repeat, len(by_kata))
    if total is not None:
        out["total"] = total
    return out


def verdict(a: Stats, b: Stats) -> str:
    """How B's mean Green compares with A's: a difference no bigger than the wider of the two ranges is noise."""
    diff = b.green - a.green
    if diff == 0:
        return "same"
    if a.n < 2 or b.n < 2:
        return f"{diff:+.1f} (one run: noise unknown)"
    if abs(diff) <= max(a.spread, b.spread):
        return f"{diff:+.1f} within noise"
    return f"{diff:+.1f} {'better' if diff > 0 else 'worse'}"


def _cell(s: Stats) -> str:
    rng = f" [{s.green_min}–{s.green_max}]" if s.n > 1 else ""
    return (f"{s.green:.1f}/{s.planned}{rng} · {s.attempts:.1f}a · {s.satisfied:.1f}s · {s.stuck:.1f} stuck · "
            f"{s.minutes:.1f}m · ${s.cost:.4f}")


LEGEND = ("Cell: mean Behaviors Green/planned [min–max over repeats] · implementation attempts (a) · Satisfied (s) · "
          "Stuck · minutes per run (m) · cost per run. A difference no bigger than the wider range is within noise.")


def label_table(lines: Sequence[Mapping[str, Any]], label: str) -> str:
    stats = aggregate(lines, label)
    if not stats:
        return f"no ledger lines labelled {label!r}"
    n = max(s.n for s in stats.values())
    rows = [f"label {label}: {n} repeat(s)", "", "| kata | runs | result |", "|---|---|---|"]
    rows += [f"| {kata} | {s.n} | {_cell(s)} |" for kata, s in stats.items()]
    return "\n".join([*rows, "", LEGEND])


def compare_table(lines: Sequence[Mapping[str, Any]], label_a: str, label_b: str) -> str:
    a, b = aggregate(lines, label_a), aggregate(lines, label_b)
    missing = [x for x, s in ((label_a, a), (label_b, b)) if not s]
    if missing:
        return f"no ledger lines labelled {', '.join(map(repr, missing))}"
    rows = [f"| kata | {label_a} | {label_b} | Green difference |", "|---|---|---|---|"]
    for kata in [*sorted((set(a) | set(b)) - {"total"}), "total"]:
        sa, sb = a.get(kata), b.get(kata)
        if sa is None or sb is None:
            rows.append(f"| {kata} | {_cell(sa) if sa else '-'} | {_cell(sb) if sb else '-'} | - |")
        else:
            rows.append(f"| {kata} | {_cell(sa)} | {_cell(sb)} | {verdict(sa, sb)} |")
    return "\n".join([*rows, "", LEGEND])


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
