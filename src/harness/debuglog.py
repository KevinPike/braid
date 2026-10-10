"""A timestamped debug log of every ``/api/ps`` poll and every state the status bar renders (M15 item 5).

A flicker seen live ("no model loaded" between two models) can be matched against the polls around it. The log is
only written while a Workflow run is active (``active``), including the model swaps before and after it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from harness.guard.ollama_client import RunningModel


class DebugLog:
    def __init__(self, path: Path, *, now: Callable[[], datetime] = datetime.now) -> None:
        self.path = path
        self._now = now
        self.active = False

    def write(self, kind: str, text: str) -> None:
        if not self.active:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(f"{self._now().isoformat(timespec='milliseconds')} {kind}: {text}\n")

    def poll(self, models: Sequence[RunningModel]) -> None:
        listing = ", ".join(f"{m.name}@{m.context_length}" for m in models) or "(nothing loaded)"
        self.write("poll", listing)

    def rendered(self, text: str) -> None:
        self.write("render", text)

    def lines(self) -> list[str]:
        return self.path.read_text().splitlines() if self.path.exists() else []
