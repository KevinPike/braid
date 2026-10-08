"""Eval results and their comparison report. Pure data and formatting: no Ollama, no Strands Evals."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from harness.guard.ollama_client import CallMetrics


@dataclass
class CallLog:
    """The measured counts of every model call made while one model is under test."""

    calls: list[CallMetrics] = field(default_factory=list)

    def add(self, metrics: CallMetrics) -> None:
        self.calls.append(metrics)


@dataclass(frozen=True)
class CaseRun:
    name: str
    prompt: str
    expected: str | None  # the tool the hand label says to call, or None for "no tool"
    first_tool: str | None
    correct: bool
    wall_s: float
    calls: tuple[CallMetrics, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "prompt": self.prompt, "expected": self.expected, "first_tool": self.first_tool,
            "correct": self.correct, "wall_s": round(self.wall_s, 3), "error": self.error,
            "calls": [
                {"prompt_eval_count": c.prompt_eval_count, "eval_count": c.eval_count, "tokens_per_second": c.tokens_per_second}
                for c in self.calls
            ],
        }


Status = Literal["ok", "skipped", "failed"]


@dataclass
class ModelResult:
    model: str
    status: Status = "ok"
    reason: str = ""  # why a model was skipped or failed
    num_ctx: int = 0
    cases: list[CaseRun] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # preflight warnings

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def correct(self) -> int:
        return sum(c.correct for c in self.cases)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def errors(self) -> int:
        return sum(bool(c.error) for c in self.cases)

    @property
    def _calls(self) -> list[CallMetrics]:
        return [m for c in self.cases for m in c.calls]

    @property
    def tps_values(self) -> list[float]:
        return [t for t in (m.tokens_per_second for m in self._calls) if t is not None]

    @property
    def mean_tps(self) -> float | None:
        return statistics.fmean(self.tps_values) if self.tps_values else None

    @property
    def median_tps(self) -> float | None:
        return statistics.median(self.tps_values) if self.tps_values else None

    @property
    def peak_prompt_tokens(self) -> int | None:
        return max((m.prompt_eval_count for m in self._calls), default=None)

    @property
    def peak_context_share(self) -> float | None:
        peak = self.peak_prompt_tokens
        return peak / self.num_ctx if peak is not None and self.num_ctx else None

    @property
    def mean_prompt_tokens(self) -> float | None:
        return statistics.fmean(m.prompt_eval_count for m in self._calls) if self._calls else None

    @property
    def wall_s(self) -> float:
        return sum(c.wall_s for c in self.cases)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model, "status": self.status, "reason": self.reason, "num_ctx": self.num_ctx,
            "cases": self.n, "correct": self.correct, "accuracy": self.accuracy, "errors": self.errors,
            "mean_tokens_per_second": self.mean_tps, "median_tokens_per_second": self.median_tps,
            "peak_prompt_tokens": self.peak_prompt_tokens, "peak_context_share": self.peak_context_share,
            "mean_prompt_tokens": self.mean_prompt_tokens, "wall_s": round(self.wall_s, 3), "notes": self.notes,
        }


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def render_markdown(results: list[ModelResult], *, when: datetime, num_ctx: int, cases_path: str) -> str:
    lines = [
        "# braid eval: tool selection per model",
        "",
        f"{when.strftime('%Y-%m-%d %H:%M:%S')} UTC, cases `{cases_path}`, num_ctx {num_ctx}.",
        "Accuracy is the share of prompts whose first tool call matched the hand label (or that called none).",
        "",
        "| model | accuracy | tok/s mean | tok/s median | peak prompt tokens | peak context use | mean prompt tokens | wall s | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    skipped: list[ModelResult] = []
    for r in results:
        if r.status != "ok":
            skipped.append(r)
            continue
        acc = "-" if r.accuracy is None else f"{r.correct}/{r.n} ({r.accuracy:.0%})"
        share = _fmt(None if r.peak_context_share is None else r.peak_context_share * 100, ".1f")
        share = share if share == "-" else f"{share}% of {r.num_ctx}"
        lines.append(
            f"| {r.model} | {acc} | {_fmt(r.mean_tps, '.1f')} | {_fmt(r.median_tps, '.1f')} | "
            f"{_fmt(r.peak_prompt_tokens, 'd')} | {share} | {_fmt(r.mean_prompt_tokens, '.0f')} | {r.wall_s:.1f} | {r.errors} |"
        )
    if skipped:
        lines += ["", "Skipped:", ""]
        lines += [f"- {r.model}: {r.status}, {r.reason}" for r in skipped]
    notes = [(r.model, n) for r in results for n in r.notes]
    if notes:
        lines += ["", "Preflight notes:", ""] + [f"- {m}: {n}" for m, n in notes]
    misses = [(r.model, c) for r in results for c in r.cases if not c.correct]
    if misses:
        lines += ["", "Misses:", ""]
        for model, c in misses:
            detail = f"error: {c.error}" if c.error else f"called {c.first_tool or 'no tool'}, expected {c.expected or 'no tool'}"
            lines.append(f"- {model}: \"{c.prompt}\" ({detail})")
    return "\n".join(lines) + "\n"


def to_json(results: list[ModelResult], *, when: datetime, num_ctx: int, cases_path: str) -> str:
    doc = {
        "when": when.isoformat(), "num_ctx": num_ctx, "cases_path": cases_path,
        "models": [{**r.summary(), "case_runs": [c.to_dict() for c in r.cases]} for r in results],
    }
    return json.dumps(doc, indent=2) + "\n"


def write_reports(
    results: list[ModelResult], evals_dir: Path, *, num_ctx: int, cases_path: str, when: datetime | None = None
) -> tuple[Path, Path, str]:
    """Write ``<UTC timestamp>.md`` and ``.json`` into ``evals_dir``; returns both paths and the markdown."""
    when = when or datetime.now(timezone.utc)
    evals_dir.mkdir(parents=True, exist_ok=True)
    stem = when.strftime("%Y%m%dT%H%M%SZ")
    md = render_markdown(results, when=when, num_ctx=num_ctx, cases_path=cases_path)
    md_path, json_path = evals_dir / f"{stem}.md", evals_dir / f"{stem}.json"
    md_path.write_text(md)
    json_path.write_text(to_json(results, when=when, num_ctx=num_ctx, cases_path=cases_path))
    return md_path, json_path, md
