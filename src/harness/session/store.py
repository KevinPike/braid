"""Session store: the SQLite record of sessions, their history, prompts, queue and exact Ollama calls.

Strands' ``FileSessionManager`` is deliberately not used. It appends messages as they are added and
has no notion of Compaction rewriting history in place (``messages[:] = ...``), so a resumed
session could replay turns the Trim log says were dropped. Here every save replaces the whole
snapshot in one transaction, which also keeps the plan's choice of SQLite for sessions.

``":memory:"`` keeps it in RAM (tests, and the app in test mode).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from strands.types.content import Message

TITLE_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    parent_id TEXT,
    model TEXT NOT NULL,
    num_ctx INTEGER NOT NULL,
    calibration TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    at TEXT NOT NULL,
    text TEXT NOT NULL,
    inherited INTEGER NOT NULL DEFAULT 0  -- copied into a fork; kept out of the global recall
);
CREATE TABLE IF NOT EXISTS queue (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (session_id, position)
);
CREATE TABLE IF NOT EXISTS calls (
    session_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    at TEXT NOT NULL,
    model TEXT NOT NULL,
    num_ctx INTEGER NOT NULL,
    payload TEXT NOT NULL,
    estimate INTEGER,
    prompt_eval_count INTEGER,
    eval_count INTEGER,
    PRIMARY KEY (session_id, n)
);
"""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SessionInfo:
    id: str
    created_at: datetime
    updated_at: datetime
    title: str
    parent_id: str | None
    model: str
    num_ctx: int
    calibration: Mapping[str, Any] = field(default_factory=dict)
    message_count: int = 0

    @property
    def estimator_state(self) -> Mapping[str, Any]:
        state = self.calibration.get("estimator")
        return state if isinstance(state, Mapping) else {}

    def line(self, *, current: bool = False) -> str:
        when = self.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
        mark = "*" if current else " "
        fork = f" (fork of {self.parent_id})" if self.parent_id else ""
        return (f"{mark} {self.id}  {self.title or '(untitled)'}{fork}  {self.model} num_ctx {self.num_ctx}  "
                f"{self.message_count} messages  {when}")


@dataclass(frozen=True)
class PromptEntry:
    id: int
    session_id: str
    at: datetime
    text: str


@dataclass(frozen=True)
class CallEntry:
    """One request to Ollama: the exact JSON body sent, with the estimate and, once done, the measured counts."""

    session_id: str
    n: int
    at: datetime
    model: str
    num_ctx: int
    payload: Mapping[str, Any]
    estimate: int | None = None  # None for calls the ledger did not estimate (summarizer)
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    @property
    def error(self) -> float | None:
        if self.estimate is None or not self.prompt_eval_count:
            return None
        return abs(self.estimate - self.prompt_eval_count) / self.prompt_eval_count

    def line(self) -> str:
        est = "-" if self.estimate is None else str(self.estimate)
        got = "…" if self.prompt_eval_count is None else str(self.prompt_eval_count)
        err = "" if self.error is None else f"  {self.error:.1%}"
        return f"#{self.n}  {self.model}  est {est}  ollama {got}{err}"


def settle(messages: Sequence[Message]) -> tuple[list[Message], int]:
    """Cut a snapshot back to its last finished turn; returns (messages, number dropped).

    A kill mid-turn leaves a user message with no reply, or a tool call with no final answer.
    Resuming into that would send two user messages in a row, so the unfinished tail is dropped
    (the prompt is still in the prompt history).
    """
    out = list(messages)
    while out and not (out[-1].get("role") == "assistant" and not any("toolUse" in b for b in out[-1].get("content", ()))):
        out.pop()
    return out, len(messages) - len(out)


class SessionStore:
    def __init__(self, path: Path | str = ":memory:", *, now: Callable[[], datetime] = _utc_now) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(_SCHEMA)
        self._now = now

    # --- sessions ---------------------------------------------------------------------------

    def create(self, model: str, num_ctx: int, calibration: Mapping[str, Any] | None = None) -> SessionInfo:
        sid = secrets.token_hex(4)
        at = self._now().isoformat()
        with self._db:
            self._db.execute(
                "INSERT INTO sessions (id, created_at, updated_at, model, num_ctx, calibration) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, at, at, model, num_ctx, json.dumps(calibration or {})),
            )
        info = self.get(sid)
        assert info is not None
        return info

    _COLUMNS = (
        "s.id, s.created_at, s.updated_at, s.title, s.parent_id, s.model, s.num_ctx, s.calibration,"
        " (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
    )

    @staticmethod
    def _info(r: Sequence[Any]) -> SessionInfo:
        return SessionInfo(r[0], datetime.fromisoformat(r[1]), datetime.fromisoformat(r[2]), r[3], r[4], r[5], r[6],
                           json.loads(r[7]), r[8])

    def get(self, session_id: str) -> SessionInfo | None:
        row = self._db.execute(f"SELECT {self._COLUMNS} FROM sessions s WHERE s.id = ?", (session_id,)).fetchone()
        return self._info(row) if row else None

    def resolve(self, prefix: str) -> SessionInfo | None:
        """A session by id or unambiguous id prefix."""
        rows = self._db.execute(f"SELECT {self._COLUMNS} FROM sessions s WHERE s.id LIKE ?", (prefix + "%",)).fetchall()
        exact = [r for r in rows if r[0] == prefix]
        rows = exact or rows
        return self._info(rows[0]) if len(rows) == 1 else None

    def sessions(self, limit: int | None = None, *, include_empty: bool = False) -> list[SessionInfo]:
        """Most recently updated first. Sessions with no messages (opened and abandoned) are hidden by default."""
        where = "" if include_empty else "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)"
        rows = self._db.execute(
            f"SELECT {self._COLUMNS} FROM sessions s {where} ORDER BY s.updated_at DESC, s.rowid DESC"
        ).fetchall()
        out = [self._info(r) for r in rows]
        return out if limit is None else out[:limit]

    def latest(self) -> SessionInfo | None:
        found = self.sessions(limit=1)
        return found[0] if found else None

    def update(self, session_id: str, *, model: str | None = None, num_ctx: int | None = None,
               calibration: Mapping[str, Any] | None = None) -> None:
        sets: list[str] = ["updated_at = ?"]
        args: list[Any] = [self._now().isoformat()]
        for col, val in (("model", model), ("num_ctx", num_ctx)):
            if val is not None:
                sets.append(f"{col} = ?")
                args.append(val)
        if calibration is not None:
            sets.append("calibration = ?")
            args.append(json.dumps(calibration))
        with self._db:
            self._db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*args, session_id))

    def _touch(self, session_id: str) -> None:
        self._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (self._now().isoformat(), session_id))

    # --- messages ---------------------------------------------------------------------------

    def save_messages(self, session_id: str, messages: Sequence[Message]) -> None:
        """Replace the whole snapshot (Compaction rewrites history in place, so appending would be wrong)."""
        rows = [(session_id, i, json.dumps(m, default=str)) for i, m in enumerate(messages)]
        with self._db:
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._db.executemany("INSERT INTO messages (session_id, idx, json) VALUES (?, ?, ?)", rows)
            self._touch(session_id)

    def messages(self, session_id: str) -> list[Message]:
        rows = self._db.execute("SELECT json FROM messages WHERE session_id = ? ORDER BY idx", (session_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    # --- prompts ----------------------------------------------------------------------------

    def add_prompt(self, session_id: str, text: str) -> None:
        with self._db:
            self._db.execute("INSERT INTO prompts (session_id, at, text) VALUES (?, ?, ?)",
                             (session_id, self._now().isoformat(), text))
            if not text.startswith("/"):  # a slash command is not what the session is about
                self._db.execute("UPDATE sessions SET title = ? WHERE id = ? AND title = ''", (_clip(text), session_id))
            self._touch(session_id)

    def prompts(self, session_id: str | None = None) -> list[PromptEntry]:
        """Oldest first. All sessions' own prompts (the global recall) by default; ``session_id`` includes inherited ones."""
        if session_id is None:
            rows = self._db.execute("SELECT id, session_id, at, text FROM prompts WHERE inherited = 0 ORDER BY id").fetchall()
        else:
            rows = self._db.execute("SELECT id, session_id, at, text FROM prompts WHERE session_id = ? ORDER BY id",
                                    (session_id,)).fetchall()
        return [PromptEntry(r[0], r[1], datetime.fromisoformat(r[2]), r[3]) for r in rows]

    # --- queue ------------------------------------------------------------------------------

    def save_queue(self, session_id: str, prompts: Sequence[str]) -> None:
        with self._db:
            self._db.execute("DELETE FROM queue WHERE session_id = ?", (session_id,))
            self._db.executemany("INSERT INTO queue (session_id, position, text) VALUES (?, ?, ?)",
                                 [(session_id, i, p) for i, p in enumerate(prompts)])

    def queue(self, session_id: str) -> list[str]:
        rows = self._db.execute("SELECT text FROM queue WHERE session_id = ? ORDER BY position", (session_id,)).fetchall()
        return [r[0] for r in rows]

    # --- calls ------------------------------------------------------------------------------

    def record_call(self, session_id: str, model: str, num_ctx: int, payload: Mapping[str, Any],
                    estimate: int | None = None) -> int:
        """Store the request body exactly as sent; returns the call number within the session."""
        with self._db:
            n = self._db.execute("SELECT COALESCE(MAX(n), 0) + 1 FROM calls WHERE session_id = ?", (session_id,)).fetchone()[0]
            self._db.execute(
                "INSERT INTO calls (session_id, n, at, model, num_ctx, payload, estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, n, self._now().isoformat(), model, num_ctx, json.dumps(payload, default=str), estimate),
            )
        return int(n)

    def complete_call(self, session_id: str, n: int, prompt_eval_count: int, eval_count: int) -> None:
        with self._db:
            self._db.execute("UPDATE calls SET prompt_eval_count = ?, eval_count = ? WHERE session_id = ? AND n = ?",
                             (prompt_eval_count, eval_count, session_id, n))

    def calls(self, session_id: str) -> list[CallEntry]:
        rows = self._db.execute(
            "SELECT session_id, n, at, model, num_ctx, payload, estimate, prompt_eval_count, eval_count"
            " FROM calls WHERE session_id = ? ORDER BY n", (session_id,)
        ).fetchall()
        return [CallEntry(r[0], r[1], datetime.fromisoformat(r[2]), r[3], r[4], json.loads(r[5]), r[6], r[7], r[8])
                for r in rows]

    # --- fork -------------------------------------------------------------------------------

    def fork(self, source_id: str) -> SessionInfo:
        """A new session with the source's model, num_ctx, calibration, messages and prompts; not its calls or queue."""
        src = self.get(source_id)
        if src is None:
            raise KeyError(source_id)
        info = self.create(src.model, src.num_ctx, src.calibration)
        with self._db:
            self._db.execute("UPDATE sessions SET parent_id = ?, title = ? WHERE id = ?", (source_id, src.title, info.id))
            self._db.execute("INSERT INTO messages (session_id, idx, json) SELECT ?, idx, json FROM messages WHERE session_id = ?",
                             (info.id, source_id))
            self._db.execute("INSERT INTO prompts (session_id, at, text, inherited) SELECT ?, at, text, 1 FROM prompts"
                             " WHERE session_id = ? ORDER BY id", (info.id, source_id))
        forked = self.get(info.id)
        assert forked is not None
        return forked

    def close(self) -> None:
        self._db.close()


def _clip(text: str) -> str:
    one = " ".join(text.split())
    return one if len(one) <= TITLE_CHARS else one[: TITLE_CHARS - 1] + "…"
