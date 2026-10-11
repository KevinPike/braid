"""The Plan: the hand-off from the Plan loop (or ``claude -p``) to the Execution loop."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from harness.workflow.invariants import Invariant


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
    invariants: list[Invariant] = []  # F3: rules every expected value in a test must satisfy, checked by code

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.model_validate_json(path.expanduser().read_text())

    def save(self, path: Path) -> None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    def to_markdown(self) -> str:
        """A readable rendering for the human who approves it; empty sections are left out."""
        out = [f"# {self.goal}", "", "## Behaviors", *(f"{i}. {b}" for i, b in enumerate(self.behaviors, 1))]

        def section(title: str, lines: list[str]) -> None:
            if lines:
                out.extend(["", f"## {title}", *lines])

        section("Assumptions", [
            f"- {a.text}" + (f" (probe: `{a.probe}`)" if a.probe else "") + (f" (evidence: {a.evidence})" if a.evidence else "")
            for a in self.assumptions
        ])
        section("Files", [f"- `{f}`" for f in self.files])
        section("Demo", [f"`{self.demo}`"] if self.demo else [])
        section("Invariants", [f"- {i.text}: `{i.check}`" + (f" (for {i.fn})" if i.fn else "") for i in self.invariants])
        section("Out of scope", [f"- {s}" for s in self.out_of_scope])
        section("Open questions", [f"- {q}" for q in self.open_questions])
        section("Notes", [f"- {n}" for n in self.notes])
        return "\n".join(out) + "\n"
