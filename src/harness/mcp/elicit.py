"""MCP elicitation through the Asker (F17, ADR 0010, 0012).

A server's ``elicitation/create`` is a question to the human, so it takes the same road as the chat tool: it becomes a
``ChoiceRequest`` (source ``mcp:<server>``), the TUI shows it and holds the Prompt queue, and the outcome maps back to
MCP's three actions. It arrives on the MCP client's background thread, so the Asker runs on the app's loop and the
client thread awaits it there. A Choice gates nothing: an elicitation never stands in for an Approval.

Only fields a Choice can show are supported: a string enum, a boolean, or a list of enum strings, each with two to four
values. Anything else (free text, numbers, a longer enum) is declined with a note, never guessed at.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from mcp import types
from pydantic import ValidationError

from harness.choice import OTHER, Asker, ChoiceRequest, ChoiceResult, ChoiceStatus, problems, validate

Action = Literal["accept", "decline", "cancel"]
# Esc is the user saying no; a cancel (Ctrl-C, a quit) or a timeout is the user not answering.
ACTIONS: dict[ChoiceStatus, Action] = {
    "answered": "accept", "declined": "decline", "cancelled": "cancel", "timed_out": "cancel", "refused": "cancel"}
YES, NO = "Yes", "No"
Value = str | int | float | bool | list[str] | None


class Unsupported(Exception):
    """The form asks for something a Choice cannot show."""


@dataclass(frozen=True)
class Field:
    name: str
    question: str  # the Question.question this field became
    values: dict[str, Any]  # option label -> the value the server expects
    multi: bool

    def decode(self, picked: tuple[str, ...], other: str | None) -> Value:
        chosen = [other if p == OTHER and other is not None else self.values.get(p, p) for p in picked]
        return [str(c) for c in chosen] if self.multi else chosen[0] if chosen else None


def _options(prop: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """(label, description, value) for an enum-shaped property, or ``Unsupported``."""
    items = prop.get("items", {}) if prop.get("type") == "array" else prop
    if prop.get("type") == "boolean":
        return [(YES, "", True), (NO, "", False)]
    if "oneOf" in items or "anyOf" in items:
        pairs = [(str(o["const"]), str(o.get("title", ""))) for o in items.get("oneOf", items.get("anyOf", [])) if "const" in o]
        return [(title or value, value if title else "", value) for value, title in pairs]
    if "enum" in items:
        names = items.get("enumNames") or items.get("x-enumNames") or []
        return [(str(names[i]) if i < len(names) else str(v), str(v) if i < len(names) else "", str(v)) for i, v in enumerate(items["enum"])]
    raise Unsupported("only a choice of two to four values, a yes/no or a list of those can be shown")


def form_request(message: str, schema: dict[str, Any]) -> tuple[ChoiceRequest, list[Field]]:
    """The Choice for an elicitation form, and how to read its answers back. ``Unsupported`` or ``ValidationError`` if it
    cannot be shown."""
    props: dict[str, Any] = schema.get("properties", {})
    if not props:
        raise Unsupported("the form has no fields")
    questions: list[dict[str, Any]] = []
    fields: list[Field] = []
    for name, prop in props.items():
        try:
            options = _options(prop)
        except Unsupported as exc:
            raise Unsupported(f"field '{name}': {exc}") from None
        label = str(prop.get("title") or name)
        text = message if len(props) == 1 else f"{message} — {label}"
        if len(props) == 1 and prop.get("description"):
            text = f"{message} ({prop['description']})"
        multi = prop.get("type") == "array"
        questions.append({
            "header": label[:12], "question": text[:300], "multiSelect": multi,
            "options": [{"label": lab[:40], "description": desc[:200]} for lab, desc, _ in options]})
        fields.append(Field(name, text[:300], {lab[:40]: value for lab, _, value in options}, multi))
    return validate({"questions": questions}, elicited=True), fields


def content(result: ChoiceResult, fields: list[Field]) -> dict[str, Value]:
    return {f.name: f.decode(a.picked, a.other) for f in fields if (a := result.answers.get(f.question)) is not None}


class Elicitor:
    """The hub's ``elicitation_callback``: ``(server, params) -> ElicitResult``.

    ``loop`` is the app's event loop (None before it runs), ``notify`` posts a note for the human. Nothing here raises
    for a user outcome or for a form that cannot be shown: both are an action the server can act on."""

    def __init__(self, asker: Asker, loop: Callable[[], asyncio.AbstractEventLoop | None], notify: Callable[[str], None] | None = None) -> None:
        self._asker, self._loop, self._notify = asker, loop, notify

    def _say(self, text: str) -> None:
        if self._notify is not None:
            self._notify(text)

    async def __call__(self, server: str, params: types.ElicitRequestParams) -> types.ElicitResult | types.ErrorData:
        if not isinstance(params, types.ElicitRequestFormParams):
            self._say(f"mcp {server}: declined a URL elicitation (only forms are supported)")
            return types.ElicitResult(action="decline")
        try:
            request, fields = form_request(params.message, dict(params.requested_schema))
        except (Unsupported, ValidationError) as exc:
            why = problems(exc) if isinstance(exc, ValidationError) else str(exc)
            self._say(f"mcp {server}: declined an elicitation: {why}")
            return types.ElicitResult(action="decline")
        loop = self._loop()
        if loop is None:
            return types.ElicitResult(action="cancel")
        async def ask() -> ChoiceResult:
            return await self._asker(request, f"mcp:{server}")

        try:  # we are on the MCP client's thread: the widget lives on the app's loop
            result = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(ask(), loop))
        except RuntimeError:  # the app loop is closed: nobody is there to ask
            return types.ElicitResult(action="cancel")
        action = ACTIONS[result.status]
        return types.ElicitResult(action=action, content=content(result, fields) if action == "accept" else None)
