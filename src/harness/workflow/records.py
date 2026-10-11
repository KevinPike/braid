"""What a run leaves behind: per-Behavior stats for the summary and a JSONL log with one line per node."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from harness.workflow.model import ApiCall

TAIL_LINES = 40


def tail(text: str, limit: int = TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-limit:])


@dataclass
class BehaviorStats:
    behavior: str
    green: bool = False
    attempts: int = 0
    red_rejections: int = 0
    hints: int = 0
    seconds: float = 0.0
    satisfied: bool = False  # the new test passed at Red and failed against the HEAD implementation: earlier code covers it
    note: str = ""
    cost_usd: float = 0.0  # API-backed runs only
    disputes: int = 0  # F3: Disputes the implementer raised on this Behavior
    disputes_upheld: int = 0  # ... that an Invariant check upheld (the test went back to write_test)


@dataclass
class Recorder:
    behaviors: list[BehaviorStats] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    stuck: int = 0
    polish: str = "not run"
    started: float = field(default_factory=time.monotonic)
    behavior_started: float = 0.0
    model_calls: int = 0
    model_seconds: float = 0.0
    cost_usd: float = 0.0  # running total over API calls (Ollama costs nothing): what the Spend cap compares
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add_usage(self, calls: Sequence[ApiCall], behavior: int | None = None) -> dict[str, Any]:
        """Count API ``calls`` toward the totals (and Behavior ``behavior``'s cost); returns the run-log fields for the
        node that made them. Empty (so the line is unchanged) when there were none, as in every Ollama run."""
        if not calls:
            return {}
        cost = sum(c.cost_usd for c in calls)
        fields: dict[str, Any] = {
            "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
            "cache_read_input_tokens": sum(c.cache_read_input_tokens for c in calls),
            "cache_creation_input_tokens": sum(c.cache_creation_input_tokens for c in calls),
            "cost_usd": cost, "request_id": calls[-1].request_id,
        }
        if len(calls) > 1:  # a retry: every request id, the last one produced the reply
            fields["request_ids"] = [c.request_id for c in calls]
        self.cost_usd += cost
        self.input_tokens += fields["input_tokens"]
        self.output_tokens += fields["output_tokens"]
        self.cache_read_input_tokens += fields["cache_read_input_tokens"]
        if behavior is not None and 0 <= behavior < len(self.behaviors):
            self.behaviors[behavior].cost_usd += cost
        return fields

    def dump(self) -> dict[str, Any]:
        """Plain data for the checkpointed state. Clocks are stored as ages, so a resumed run (a new process, a
        different monotonic origin) keeps counting its elapsed and per-Behavior seconds from where they were."""
        now = time.monotonic()
        return {
            "behaviors": [asdict(b) for b in self.behaviors], "trace": list(self.trace), "stuck": self.stuck,
            "polish": self.polish, "elapsed_s": now - self.started,
            "behavior_age_s": now - self.behavior_started if self.behavior_started else 0.0,
            "model_calls": self.model_calls, "model_seconds": self.model_seconds, "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    def load(self, data: dict[str, Any]) -> None:
        """Replace this Recorder's contents with a ``dump()``; the run goes on as if it had never stopped."""
        now = time.monotonic()
        self.behaviors[:] = [BehaviorStats(**b) for b in data["behaviors"]]
        self.trace[:] = data["trace"]
        self.stuck, self.polish = data["stuck"], data["polish"]
        self.started = now - data["elapsed_s"]
        self.behavior_started = now - data["behavior_age_s"] if data["behavior_age_s"] else 0.0
        self.model_calls, self.model_seconds = data["model_calls"], data["model_seconds"]
        self.cost_usd = data.get("cost_usd", 0.0)  # absent in a checkpoint from before the API backend
        self.input_tokens, self.output_tokens = data.get("input_tokens", 0), data.get("output_tokens", 0)
        self.cache_read_input_tokens = data.get("cache_read_input_tokens", 0)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started


class UsageMeter:
    """Hands out the API calls a model made since the last ``take``, so each node's log line carries its own usage.
    A model without ``ApiCall``s in a ``calls`` list (anything Ollama-backed, or a fake) has none."""

    def __init__(self, model: object) -> None:
        calls = getattr(model, "calls", None)
        self.model = model
        self.mark = len(calls) if isinstance(calls, list) else 0  # a model reused across runs: only this run's calls

    def take(self) -> list[ApiCall]:
        calls = getattr(self.model, "calls", None)
        if not isinstance(calls, list):
            return []
        new = calls[self.mark:]
        self.mark = len(calls)
        return [c for c in new if isinstance(c, ApiCall)]


class RunLog:
    """Appends one JSON object per node to ``path``; rich enough to debug a Behavior without rerunning it."""

    def __init__(self, path: Path, listener: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path = path
        self.listener = listener  # sees each record as written: the CLI's progress lines come from the same numbers
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, node: str, **fields: Any) -> None:
        line = {"node": node, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        if self.listener:
            self.listener(json.loads(json.dumps(line, default=str)))

    def lines(self) -> list[dict[str, Any]]:
        return read_log(self.path)


def read_log(path: Path) -> list[dict[str, Any]]:
    """The records of a run log, oldest first; empty when there is no log."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_summary(
    directory: Path, *, model: str, project: Path, no_hints: bool, status: str | None, planned: int, rec: Recorder,
    workspace: Path, run_log: Path | None, now: datetime | None = None, provenance: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """The M11 summary JSON (``harness-workflow report`` reads it): per Behavior and in total, plus F2's provenance
    (``workflow/bench.py``: which harness commit, Plan, backend and settings produced it). ``name`` defaults to a
    timestamp and the model."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{name or stamp + '-' + re.sub(r'[^A-Za-z0-9._-]', '_', model)}.json"
    b = rec.behaviors
    path.write_text(json.dumps({
        "model": model, "kata": project.name, "no_hints": no_hints, "status": status,
        "behaviors": [asdict(x) for x in b], "planned": planned,
        "totals": {
            "green": sum(x.green for x in b), "attempts": sum(x.attempts for x in b),
            "red_rejections": sum(x.red_rejections for x in b), "hints": sum(x.hints for x in b),
            "satisfied": sum(x.satisfied for x in b), "seconds": round(time.monotonic() - rec.started, 1),
            "disputes": sum(x.disputes for x in b), "disputes_upheld": sum(x.disputes_upheld for x in b),
            "model_seconds": round(rec.model_seconds, 1), "model_calls": rec.model_calls,
            "cost_usd": round(rec.cost_usd, 8), "input_tokens": rec.input_tokens, "output_tokens": rec.output_tokens,
            "cache_read_input_tokens": rec.cache_read_input_tokens,
        },
        "stuck": rec.stuck, "polish": rec.polish, "trace": rec.trace, "workspace": str(workspace),
        "run_log": str(run_log) if run_log else None,
        **(provenance or {}),
    }, indent=2) + "\n")
    return path
