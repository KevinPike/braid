"""What a run leaves behind: per-Behavior stats for the summary and a JSONL log with one line per node, and one Route
record per conditional edge taken (F9, ADR 0008)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from harness.workflow.model import ApiCall

TAIL_LINES = 40
VALUE_CHARS = 200  # a Route record's value is cut to this (a hint or a pytest tail can be long)

Basis = Literal["code", "model", "human"]
BASES: tuple[Basis, ...] = ("code", "model", "human")


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
    arbitrations: int = 0  # F10: Disputes no Invariant upheld that went to the Arbiter
    rulings_test_wrong: int = 0  # ... and its rulings: the test went back to write_test
    rulings_test_right: int = 0  # ... the test stands, and the ruling went into the next write_impl prompt


@dataclass(frozen=True)
class Route:
    """Where a conditional edge goes, and why: a Route record without its source (ADR 0008). ``field`` is the state
    field that decided it and ``value`` its value; a fixed edge, which only the Spend cap can divert, has neither.

    The ``after_*`` route functions return one, and the loop's ``gate`` writes it to the run log. A model basis (F10) is
    built with ``Route.by_model``: the field is a schema field of a node's structured reply, and the record also carries
    the model, the API request id, the claim's reason and the thinking summary."""

    to: str
    field: str | None = None
    value: Any = None
    basis: Basis = "code"
    claim: Mapping[str, Any] | None = None  # model basis only: model, request_id, reason, thinking
    sends: tuple[Any, ...] = ()  # a fan-out: the ``Send``s LangGraph is given instead of ``to``

    @classmethod
    def by_model(
        cls, to: str, field: str, value: Any, *, model: str, request_id: str | None, reason: str, thinking: str = "",
    ) -> Route:
        return cls(to, field, value, "model", {"model": model, "request_id": request_id, "reason": reason, "thinking": thinking})

    def record(self, source: str) -> dict[str, Any]:
        """The run-log fields: ``{from, to, basis, field, value}``, plus the claim for a model basis."""
        value = self.value
        if isinstance(value, str) and len(value) > VALUE_CHARS:
            value = value[:VALUE_CHARS] + "…"
        return {"from": source, "to": self.to, "basis": self.basis, "field": self.field, "value": value, **(self.claim or {})}


def route_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Route records by basis, as the summary's ``routes`` has them."""
    counts: dict[str, int] = dict.fromkeys(BASES, 0)
    for r in records:
        if r.get("node") == "route":
            counts[r["basis"]] = counts.get(r["basis"], 0) + 1
    return counts


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
    routes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BASES, 0))  # Route records by basis (F9)
    roles: dict[str, dict[str, float]] = field(default_factory=dict)  # Role -> {calls, cost_usd}, API calls only (F10)
    proposed_invariants: list[dict[str, Any]] = field(default_factory=list)  # F10: from rulings, for a human to add

    def add_route(self, route: Route) -> None:
        self.routes[route.basis] = self.routes.get(route.basis, 0) + 1

    def add_role(self, role: str, calls: Sequence[ApiCall]) -> None:
        """Count ``calls`` under ``role`` (tester, implementer, arbiter): the summary reports cost per Role."""
        if calls:
            tally = self.roles.setdefault(role, {"calls": 0, "cost_usd": 0.0})
            tally["calls"] += len(calls)
            tally["cost_usd"] += sum(c.cost_usd for c in calls)

    def add_usage(self, calls: Sequence[ApiCall], behavior: int | None = None) -> dict[str, Any]:
        """Count API ``calls`` toward the totals (and Behavior ``behavior``'s cost); returns the run-log fields for the
        node that made them. Empty (so the line is unchanged) when there were none, as in every Ollama run. ``thinking``
        holds the replies' thinking summaries (F9), when the API sent any."""
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
        if thinking := "\n\n".join(c.thinking for c in calls if c.thinking):
            fields["thinking"] = thinking
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
            "cache_read_input_tokens": self.cache_read_input_tokens, "routes": dict(self.routes),
            "roles": {k: dict(v) for k, v in self.roles.items()}, "proposed_invariants": list(self.proposed_invariants),
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
        self.routes = {**dict.fromkeys(BASES, 0), **data.get("routes", {})}  # absent before F9
        self.roles = {k: dict(v) for k, v in data.get("roles", {}).items()}  # absent before F10
        self.proposed_invariants = list(data.get("proposed_invariants", []))

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
        new = self.peek()
        calls = getattr(self.model, "calls", None)
        self.mark = len(calls) if isinstance(calls, list) else self.mark
        return new

    def peek(self) -> list[ApiCall]:
        """The calls ``take`` would hand out, left in place: a node reads its own request id and thinking (F10)."""
        calls = getattr(self.model, "calls", None)
        if not isinstance(calls, list):
            return []
        return [c for c in calls[self.mark:] if isinstance(c, ApiCall)]


class RunLog:
    """Appends one JSON object per node to ``path``; rich enough to debug a Behavior without rerunning it."""

    def __init__(self, path: Path, listener: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path = path
        self.listener = listener  # sees each record as written: the CLI's progress lines come from the same numbers
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, node: str, **fields: Any) -> None:
        line = self._append(node, fields)
        if self.listener:
            self.listener(json.loads(json.dumps(line, default=str)))

    def route(self, source: str, route: Route, **fields: Any) -> None:
        """A Route record. It goes to the log only: the listener's progress lines are one per node."""
        self._append("route", {**fields, **route.record(source)})

    def _append(self, node: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        line = {"node": node, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        return line

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
    timestamp and the model. ``routes`` counts the run log's Route records by basis (F9): the log, because a resumed
    Recorder comes from a Checkpoint taken before the last node's route was counted."""
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
            "arbitrations": sum(x.arbitrations for x in b), "rulings_test_wrong": sum(x.rulings_test_wrong for x in b),
            "rulings_test_right": sum(x.rulings_test_right for x in b),
            "model_seconds": round(rec.model_seconds, 1), "model_calls": rec.model_calls,
            "cost_usd": round(rec.cost_usd, 8), "input_tokens": rec.input_tokens, "output_tokens": rec.output_tokens,
            "cache_read_input_tokens": rec.cache_read_input_tokens,
        },
        "routes": route_counts(read_log(run_log)) if run_log else dict(rec.routes),
        "roles": {k: {"calls": int(v["calls"]), "cost_usd": round(v["cost_usd"], 8)} for k, v in rec.roles.items()},
        "proposed_invariants": rec.proposed_invariants,
        "stuck": rec.stuck, "polish": rec.polish, "trace": rec.trace, "workspace": str(workspace),
        "run_log": str(run_log) if run_log else None,
        **(provenance or {}),
    }, indent=2) + "\n")
    return path


def amend_summary(path: Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Add ``fields`` to a written summary and return it: F7's hidden-test grade (``hidden_passed``, ``hidden_total``,
    ``mutants_killed``, ``mutants_total``), which is known only after the run. Summaries without them still load."""
    summary: dict[str, Any] = {**json.loads(path.read_text()), **fields}
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
