"""``ask_user_question``: the chat agent puts a Choice to the human (ADR 0010).

A hand-written Strands tool, so the JSON schema the model sees is the Pydantic one (``multiSelect``, limits and all)
rather than one inferred from a signature. It acts on nothing and needs no Approval; the answer comes back as data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from strands.types._events import ToolResultEvent
from strands.types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

from harness.choice import Asker, ChoiceResult, problems, tool_schema, validate

TOOL_NAME = "ask_user_question"
LIMIT_REASON = "ask limit for this turn reached; decide or stop"
DESCRIPTION = (
    "Ask the user one to four multiple-choice questions when a decision belongs to them: their own preferences, a choice "
    "between approaches only they can make, or which of several things they mean. Each question has two to four options "
    "(a short label, an optional description, an optional code or ASCII preview) and the user may also type their own "
    "answer (an 'Other' option is added automatically, so never include one). Set multiSelect to let them pick several. "
    "Do not use it to ask permission, to confirm a plan, or to ask 'shall I proceed?': decide and act, or ask a real "
    "question. Never ask for a password, key or token. Ask once per message, and only when you cannot reasonably decide "
    "yourself. The result maps each question's text to the chosen label (or the user's own text), with optional notes."
)


@dataclass(frozen=True)
class AskLimits:
    calls_per_turn: int = 3
    refusals_per_turn: int = 2


class AskTool(AgentTool):
    """The tool. ``asker`` shows the Choice; ``turn`` says which user turn we are in (the limits count per turn) and
    ``source`` who is asking (``chat`` or ``skill:<name>``). ``on_refused`` hears every call turned away, shown or not."""

    def __init__(
        self, asker: Asker, limits: AskLimits | None = None, *, turn: Callable[[], int] = lambda: 0,
        source: Callable[[], str] = lambda: "chat", on_refused: Callable[[Any, str, str], None] | None = None,
    ) -> None:
        super().__init__()
        self._asker, self._limits, self._turn, self._source = asker, limits or AskLimits(), turn, source
        self._on_refused = on_refused  # (raw input, source, reason): the app writes it to the Choice log
        self._spec: ToolSpec = {"name": TOOL_NAME, "description": DESCRIPTION, "inputSchema": {"json": tool_schema()}}
        self._seen_turn: int | None = None
        self._asked = 0
        self._refused = 0
        self._open = False  # a Choice is being asked right now (or waits for the human lock)

    @property
    def tool_name(self) -> str:
        return TOOL_NAME

    @property
    def tool_spec(self) -> ToolSpec:
        return self._spec

    @property
    def tool_type(self) -> str:
        return "python"

    def _refuse(self, reason: str, raw: Any = None) -> ChoiceResult:
        self._refused += 1
        if self._on_refused is not None:
            self._on_refused(raw, self._source(), reason)
        return ChoiceResult(status="refused", reason=reason)

    async def ask(self, raw: Any) -> ChoiceResult:
        """One call: refuse it, or show the Choice. A user outcome (declined, cancelled, timed out) is a result."""
        if self._turn() != self._seen_turn:  # a new user turn: the limits start again
            self._seen_turn, self._asked, self._refused, self._open = self._turn(), 0, 0, False
        if self._refused >= self._limits.refusals_per_turn:
            if self._on_refused is not None:  # not counted again and not shown, but still a row
                self._on_refused(raw, self._source(), LIMIT_REASON)
            return ChoiceResult(status="refused", reason=LIMIT_REASON)
        if self._asked >= self._limits.calls_per_turn:
            return self._refuse(LIMIT_REASON, raw)
        if self._open:
            return self._refuse("one Choice at a time: ask once per message, and wait for the answer", raw)
        try:
            request = validate(raw)
        except ValidationError as exc:
            return self._refuse(problems(exc), raw)
        self._asked += 1
        self._open = True
        try:
            return await self._asker(request, self._source())
        finally:
            self._open = False

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        result = await self.ask(tool_use.get("input", {}))
        out: ToolResult = {
            "toolUseId": tool_use["toolUseId"], "status": "error" if result.status == "refused" else "success",
            "content": [{"text": json.dumps(result.for_model())}],
        }
        yield ToolResultEvent(out)


def build_ask_tool(
    asker: Asker, limits: AskLimits | None = None, *, turn: Callable[[], int] = lambda: 0,
    source: Callable[[], str] = lambda: "chat", on_refused: Callable[[Any, str, str], None] | None = None,
) -> AskTool:
    return AskTool(asker, limits, turn=turn, source=source, on_refused=on_refused)
