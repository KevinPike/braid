"""Run bookkeeping for the TUI (M15): the checkpoint database and an index of runs beside it.

LangGraph's SQLite saver keeps each run's state per ``thread_id`` (= the run id) in ``~/.braid/workflows.db``; the
``runs`` table in the same file records what a resume needs to rebuild the graph (project, Workspace, model, Plan) and
what ``/runs`` shows (state, goal, times). Nothing here knows about Textual.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# running: a node is working; waiting: stopped for a human; paused: cancelled with Ctrl-C; finished; failed.
# A row still "running" or "waiting" that no live run owns was cut off by a kill: it resumes like a paused one.
STATES = ("running", "waiting", "paused", "finished", "failed")


@dataclass
class RunInfo:
    run_id: str
    kind: str  # "build" or "plan"
    goal: str
    project: str
    model: str
    state: str = "running"
    outcome: str = ""  # applied, kept, aborted, approved, round_limit, ...
    plan_path: str = ""  # build: the Plan file it was started from
    workspace: str = ""  # build: the Workspace directory
    hints: bool = True
    started: str = ""
    updated: str = ""
    cost_usd: float = 0.0  # spent so far (API-backed runs); kept up to date by runs started over MCP
    cap_usd: float = 0.0  # the Spend cap the run was granted; 0 for a run that never had one recorded
    origin: str = ""  # "mcp" for a run `harness-mcp` started (ADR 0013); the daily ceiling adds these up

    def line(self, *, live: bool) -> str:
        state = self.state if live or self.state not in ("running", "waiting") else "killed"
        resumable = state in ("paused", "killed", "waiting")
        tail = f" -> {self.outcome}" if self.outcome else ""
        goal = self.goal.replace("\n", " ")
        return (f"{self.run_id}  {self.kind:<5} {state:<8} {self.model}  {goal[:50]}{tail}"
                + ("  (resume: /" + f"{self.kind} --resume {self.run_id})" if resumable and state != "waiting" else ""))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class RunStore:
    def __init__(self, path: Path, *, timeout: float = 30.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, timeout=timeout)
        names = [f.name for f in fields(RunInfo)]
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, kind TEXT, goal TEXT, project TEXT, model TEXT, "
            "state TEXT, outcome TEXT, plan_path TEXT, workspace TEXT, hints INTEGER, started TEXT, updated TEXT, "
            "cost_usd REAL DEFAULT 0, cap_usd REAL DEFAULT 0, origin TEXT DEFAULT '')"
        )
        have = {row[1] for row in self._db.execute("PRAGMA table_info(runs)")}
        for name, ddl in (("cost_usd", "REAL DEFAULT 0"), ("cap_usd", "REAL DEFAULT 0"), ("origin", "TEXT DEFAULT ''")):
            if name not in have:  # an index from before the MCP server
                self._db.execute(f"ALTER TABLE runs ADD COLUMN {name} {ddl}")
        self._db.commit()
        self._names = names

    def close(self) -> None:
        self._db.close()

    def create(self, info: RunInfo) -> RunInfo:
        info.started = info.updated = _now()
        marks = ",".join("?" for _ in self._names)
        self._write(f"INSERT OR REPLACE INTO runs ({','.join(self._names)}) VALUES ({marks})", [getattr(info, n) for n in self._names])
        return info

    def update(self, run_id: str, **changes: Any) -> None:
        changes["updated"] = _now()
        sets = ", ".join(f"{k} = ?" for k in changes)
        self._write(f"UPDATE runs SET {sets} WHERE run_id = ?", [*changes.values(), run_id])

    def _write(self, sql: str, params: list[Any]) -> None:
        try:
            self._db.execute(sql, params)
            self._db.commit()
        except sqlite3.Error:
            self._db.rollback()  # else the transaction a locked write began stays open, and its stale snapshot keeps every retry failing
            raise

    def _row(self, row: tuple[Any, ...]) -> RunInfo:
        data = dict(zip(self._names, row, strict=True))
        data["hints"] = bool(data["hints"])
        return RunInfo(**data)

    def get(self, run_id: str) -> RunInfo | None:
        # fetchall, not fetchone: a statement left unfinished keeps a read snapshot open, and the next write then fails with
        # "database is locked" the moment the checkpointer (same file, another connection) has committed
        rows = self._db.execute(f"SELECT {','.join(self._names)} FROM runs WHERE run_id = ?", [run_id]).fetchall()
        return self._row(rows[0]) if rows else None

    def resolve(self, prefix: str) -> RunInfo | None:
        """The run whose id is ``prefix`` or the only one starting with it."""
        exact = self.get(prefix)
        if exact is not None:
            return exact
        rows = self._db.execute(
            f"SELECT {','.join(self._names)} FROM runs WHERE run_id LIKE ? ESCAPE '\\'",
            [prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"],
        ).fetchall()
        return self._row(rows[0]) if len(rows) == 1 else None

    def spent_since(self, since: str, *, origin: str, excluding: str | None = None) -> float:
        """What runs of ``origin`` started at or after ``since`` (an ISO UTC time) have cost, or may still cost: a run that
        is running counts at the larger of its cost so far and its Spend cap, so concurrent grants cannot overshoot."""
        rows = self._db.execute(
            "SELECT COALESCE(SUM(CASE WHEN state = 'running' THEN MAX(cost_usd, cap_usd) ELSE cost_usd END), 0) "
            "FROM runs WHERE origin = ? AND started >= ? AND run_id != ?", [origin, since, excluding or ""],
        ).fetchall()
        return float(rows[0][0])

    def list(self, limit: int = 20) -> list[RunInfo]:
        rows = self._db.execute(
            f"SELECT {','.join(self._names)} FROM runs ORDER BY updated DESC, run_id DESC LIMIT ?", [limit]).fetchall()
        return [self._row(r) for r in rows]


@contextlib.asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """The SQLite saver on ``path``, closed when the run ends (or is cancelled)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
