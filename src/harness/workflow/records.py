"""What a run leaves behind: per-Behavior stats for the summary and a JSONL log with one line per node."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

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

    def dump(self) -> dict[str, Any]:
        """Plain data for the checkpointed state. Clocks are stored as ages, so a resumed run (a new process, a
        different monotonic origin) keeps counting its elapsed and per-Behavior seconds from where they were."""
        now = time.monotonic()
        return {
            "behaviors": [asdict(b) for b in self.behaviors], "trace": list(self.trace), "stuck": self.stuck,
            "polish": self.polish, "elapsed_s": now - self.started,
            "behavior_age_s": now - self.behavior_started if self.behavior_started else 0.0,
            "model_calls": self.model_calls, "model_seconds": self.model_seconds,
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

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started


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
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []


def write_summary(
    directory: Path, *, model: str, project: Path, no_hints: bool, status: str | None, planned: int, rec: Recorder,
    workspace: Path, run_log: Path | None, now: datetime | None = None,
) -> Path:
    """The M11 summary JSON (``examples/graphs/report.py`` reads it): per Behavior and in total."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.json"
    b = rec.behaviors
    path.write_text(json.dumps({
        "model": model, "kata": project.name, "no_hints": no_hints, "status": status,
        "behaviors": [asdict(x) for x in b], "planned": planned,
        "totals": {
            "green": sum(x.green for x in b), "attempts": sum(x.attempts for x in b),
            "red_rejections": sum(x.red_rejections for x in b), "hints": sum(x.hints for x in b),
            "satisfied": sum(x.satisfied for x in b), "seconds": round(time.monotonic() - rec.started, 1),
            "model_seconds": round(rec.model_seconds, 1), "model_calls": rec.model_calls,
        },
        "stuck": rec.stuck, "polish": rec.polish, "trace": rec.trace, "workspace": str(workspace),
        "run_log": str(run_log) if run_log else None,
    }, indent=2) + "\n")
    return path
