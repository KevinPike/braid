"""Decision log: every tev1 question, the state it saw and the probabilities it returned."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    state TEXT NOT NULL,
    summarized INTEGER NOT NULL,
    input_tokens INTEGER,
    answers TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class Decision:
    kind: str  # router | compaction | ...
    model: str
    state: str
    answers: dict[str, dict[str, object]]
    summarized: bool = False
    input_tokens: int | None = None
    error: str = ""
    at: datetime | None = None
    id: int | None = None

    def line(self) -> str:
        when = self.at.astimezone().strftime("%H:%M:%S") if self.at else "--:--:--"
        head = f"#{self.id} {when} {self.kind}"
        if self.error:
            return f"{head}: failed ({self.error})"
        parts = []
        for name, a in self.answers.items():
            if "choice" in a:
                probs = a.get("probabilities") or {}
                p = probs.get(a["choice"], 0.0) if isinstance(probs, dict) else 0.0
                parts.append(f"{name}={a['choice']} ({p:.2f})")
            elif "noul" in a:
                parts.append(f"{name}=true@{float(a['noul']):.2f}")  # type: ignore[arg-type]
            elif "score" in a:
                parts.append(f"{name}=score {float(a['score']):.2f}")  # type: ignore[arg-type]
        text = f"{head}: {', '.join(parts)}"
        if self.summarized:
            text += " [state summarized]"
        preview = self.state.replace("\n", " ")
        return f"{text} | {preview[:60]}{'…' if len(preview) > 60 else ''}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionLog:
    def __init__(self, path: Path | str = ":memory:", *, now: Callable[[], datetime] = _utc_now) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(_SCHEMA)
        self._now = now
        self._listeners: list[Callable[[Decision], None]] = []

    def subscribe(self, listener: Callable[[Decision], None]) -> None:
        self._listeners.append(listener)

    def record(self, d: Decision) -> Decision:
        at = d.at or self._now()
        cur = self._db.execute(
            "INSERT INTO decisions (at, kind, model, state, summarized, input_tokens, answers, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (at.isoformat(), d.kind, d.model, d.state, int(d.summarized), d.input_tokens, json.dumps(d.answers), d.error),
        )
        self._db.commit()
        saved = Decision(d.kind, d.model, d.state, d.answers, d.summarized, d.input_tokens, d.error, at, cur.lastrowid)
        for listener in self._listeners:
            listener(saved)
        return saved

    def entries(self, limit: int | None = None) -> list[Decision]:
        rows = self._db.execute(
            "SELECT id, at, kind, model, state, summarized, input_tokens, answers, error FROM decisions ORDER BY id"
        ).fetchall()
        out = [
            Decision(r[2], r[3], r[4], json.loads(r[7]), bool(r[5]), r[6], r[8], datetime.fromisoformat(r[1]), r[0])
            for r in rows
        ]
        return out if limit is None else out[-limit:]

    def close(self) -> None:
        self._db.close()
