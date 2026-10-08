"""Trim log: the SQLite record of what compaction dropped, summarized or offloaded, when and why."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    reason TEXT NOT NULL,
    messages_removed INTEGER NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after INTEGER NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class TrimEntry:
    strategy: str  # sliding_window | summarize | offload
    reason: str
    messages_removed: int
    tokens_before: int
    tokens_after: int
    detail: str = ""
    at: datetime | None = None
    id: int | None = None

    @property
    def tokens_freed(self) -> int:
        return self.tokens_before - self.tokens_after

    def line(self) -> str:
        when = self.at.astimezone().strftime("%H:%M:%S") if self.at else "--:--:--"
        text = (
            f"#{self.id} {when} {self.strategy}: {self.reason}; "
            f"{self.messages_removed} messages, {self.tokens_before} -> {self.tokens_after} tokens"
        )
        return f"{text} ({self.detail})" if self.detail else text


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrimLog:
    """Append-only; ``":memory:"`` keeps it in RAM (tests, and sessions without a data dir)."""

    def __init__(self, path: Path | str = ":memory:", *, now: Callable[[], datetime] = _utc_now) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(_SCHEMA)
        self._now = now
        self._listeners: list[Callable[[TrimEntry], None]] = []

    def subscribe(self, listener: Callable[[TrimEntry], None]) -> None:
        self._listeners.append(listener)

    def record(self, entry: TrimEntry) -> TrimEntry:
        at = entry.at or self._now()
        cur = self._db.execute(
            "INSERT INTO trims (at, strategy, reason, messages_removed, tokens_before, tokens_after, detail)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (at.isoformat(), entry.strategy, entry.reason, entry.messages_removed,
             entry.tokens_before, entry.tokens_after, entry.detail),
        )
        self._db.commit()
        saved = TrimEntry(entry.strategy, entry.reason, entry.messages_removed, entry.tokens_before,
                          entry.tokens_after, entry.detail, at, cur.lastrowid)
        for listener in self._listeners:
            listener(saved)
        return saved

    def entries(self, limit: int | None = None) -> list[TrimEntry]:
        """Oldest first; with ``limit``, the most recent ``limit`` entries."""
        rows = self._db.execute(
            "SELECT id, at, strategy, reason, messages_removed, tokens_before, tokens_after, detail"
            " FROM trims ORDER BY id"
        ).fetchall()
        out = [
            TrimEntry(r[2], r[3], r[4], r[5], r[6], r[7], datetime.fromisoformat(r[1]), r[0]) for r in rows
        ]
        return out if limit is None else out[-limit:]

    def close(self) -> None:
        self._db.close()
