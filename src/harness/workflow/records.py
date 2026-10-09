"""What a run leaves behind: per-Behavior stats for the summary and a JSONL log with one line per node."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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


class RunLog:
    """Appends one JSON object per node to ``path``; rich enough to debug a Behavior without rerunning it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, node: str, **fields: Any) -> None:
        line = {"node": node, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    def lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
