"""The Plan: the hand-off from the Plan loop (or ``claude -p``) to the Execution loop."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Assumption(BaseModel):
    text: str
    probe: str | None = None  # shell command, run in the Sandbox
    evidence: str | None = None  # "path:line - what it shows", from Plan-loop research


class Plan(BaseModel):
    goal: str
    behaviors: list[str] = Field(min_length=1)
    assumptions: list[Assumption] = []
    files: list[str] = []  # paths the work will read or touch; seeds model context
    demo: str | None = None  # command whose output is shown at Sign-off
    out_of_scope: list[str] = []
    open_questions: list[str] = []
    notes: list[str] = []

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.model_validate_json(path.expanduser().read_text())

    def save(self, path: Path) -> None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
