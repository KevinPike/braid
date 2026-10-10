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
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, timeout=30)
        names = [f.name for f in fields(RunInfo)]
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, kind TEXT, goal TEXT, project TEXT, model TEXT, "
            "state TEXT, outcome TEXT, plan_path TEXT, workspace TEXT, hints INTEGER, started TEXT, updated TEXT)"
        )
        self._db.commit()
        self._names = names

    def close(self) -> None:
        self._db.close()

    def create(self, info: RunInfo) -> RunInfo:
        info.started = info.updated = _now()
        marks = ",".join("?" for _ in self._names)
        self._db.execute(f"INSERT OR REPLACE INTO runs ({','.join(self._names)}) VALUES ({marks})", [getattr(info, n) for n in self._names])
        self._db.commit()
        return info

    def update(self, run_id: str, **changes: Any) -> None:
        changes["updated"] = _now()
        sets = ", ".join(f"{k} = ?" for k in changes)
        self._db.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", [*changes.values(), run_id])
        self._db.commit()

    def _row(self, row: tuple[Any, ...]) -> RunInfo:
        data = dict(zip(self._names, row, strict=True))
        data["hints"] = bool(data["hints"])
        return RunInfo(**data)

    def get(self, run_id: str) -> RunInfo | None:
        row = self._db.execute(f"SELECT {','.join(self._names)} FROM runs WHERE run_id = ?", [run_id]).fetchone()
        return self._row(row) if row else None

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
