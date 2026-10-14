"""F17 gate 1: a fake server's ``ctx.elicit`` shows the TUI question (source ``mcp:<server>``), holds the Prompt queue, and
the answer returns ``accept`` with the content; Esc returns ``decline``; a timeout returns ``cancel``. The server runs in
memory, so the elicitation arrives on the MCP client's background thread as it does in life."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp import types
from mcp.server.elicitation import AcceptedElicitation, ElicitationResult
from mcp.server.mcpserver import Elicit, MCPServer, Resolve
import pytest
from pydantic import BaseModel, ValidationError
from textual.pilot import Pilot
from textual.widgets import Input

from harness.app import HarnessApp
from harness.chat import ChoicePrompt
from harness.mcp.elicit import Elicitor, Unsupported, form_request
from tests.anthropic_fakes import FakeApi, text, tool_use
from tests.mcp.helpers import in_memory
from tests.mcp.test_app import mcp_app
from tests.skills.helpers import submit, until


class Kata(BaseModel):
    kata: Literal["word-wrap", "fizzbuzz"]


class Free(BaseModel):
    name: str


def which_kata() -> Elicit[Kata]:
    return Elicit("Which kata?", Kata)


def your_name() -> Elicit[Free]:
    return Elicit("Your name?", Free)


async def pick(got: Annotated[ElicitationResult[Kata], Resolve(which_kata)]) -> str:
    """Ask the user which kata."""
    return f"{got.action}:{got.data.kata}" if isinstance(got, AcceptedElicitation) else f"{got.action}:None"


async def name(got: Annotated[ElicitationResult[Free], Resolve(your_name)]) -> str:
    """Ask for free text."""
    return got.action


def asking_server() -> MCPServer:
    """Tools that elicit. This SDK (the 2026-07-28 protocol) forbids a server's mid-call ``ctx.elicit`` request on a modern
    connection and carries the question in an ``InputRequiredResult`` instead; ``Resolve`` + ``Elicit`` is the server
    API for that, and the client answers through the same ``elicitation_callback``."""
    server = MCPServer("fake")
    server.add_tool(pick, name="pick", description="Ask the user which kata.")
    server.add_tool(name, name="name", description="Ask for free text.")
    return server


def asking_app(tmp_path: Path, api: FakeApi, *, timeout_s: float = 0.0, call_timeout_s: float | None = None) -> HarnessApp:
    app = mcp_app(tmp_path, api, allow=["pick", "name"])
    assert app.mcp is not None
    app.mcp._transports["fake"] = in_memory(asking_server())  # noqa: SLF001
    app.config.ask = app.config.ask.model_copy(update={"timeout_s": timeout_s})
    if call_timeout_s is not None:
        app.mcp.servers["fake"].cfg.timeout_s = call_timeout_s
    return app


def tool_result(api: FakeApi) -> str:
    [text_] = [str(c["text"]) for m in api.bodies[-1]["messages"] for b in m["content"]
               if isinstance(b, dict) and b["type"] == "tool_result" for c in b["content"]]
    return text_


async def open_choice(app: HarnessApp, pilot: Pilot[None]) -> ChoicePrompt:
    await until(pilot, lambda: bool(app.query(ChoicePrompt)))
    await pilot.pause(0.05)
    return app.query_one(ChoicePrompt)


async def test_an_answer_returns_accept_with_the_content_and_the_widget_holds_the_prompt_queue(tmp_path: Path) -> None:
    api = FakeApi().reply(tool_use("mcp__fake__pick", {}), stop_reason="tool_use").reply(text("ok")).reply(text("later"))
    app = asking_app(tmp_path, api)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "pick a kata")
        prompt = await open_choice(app, pilot)
        assert prompt.border_title == "Choice from mcp:fake" and prompt.source == "mcp:fake"
        box = app.query_one("#prompt", Input)  # the Choice has the focus, so Enter would answer it: submit directly
        box.value = "queued while it is open"
        await box.action_submit()
        await pilot.pause(0.05)
        assert app.queue.pending == ("queued while it is open",) and app._choices_waiting == 1
        await pilot.press("2", "enter")
        await until(pilot, lambda: not app.generating and not app.queue and len(api.bodies) == 3)
        [row] = app.choices.entries()
    assert (row.source, row.model, row.result.status) == ("mcp:fake", "mcp:fake", "answered")
    assert api.bodies[1]["messages"][-1]["content"][0]["content"][0]["text"] == "accept:fizzbuzz"


async def test_esc_returns_decline(tmp_path: Path) -> None:
    api = FakeApi().reply(tool_use("mcp__fake__pick", {}), stop_reason="tool_use").reply(text("ok"))
    app = asking_app(tmp_path, api)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "pick a kata")
        await open_choice(app, pilot)
        await pilot.press("escape")
        await until(pilot, lambda: not app.generating)
        [row] = app.choices.entries()
    assert tool_result(api) == "decline:None" and row.result.status == "declined"


async def test_a_timeout_returns_cancel_and_picks_nothing(tmp_path: Path) -> None:
    api = FakeApi().reply(tool_use("mcp__fake__pick", {}), stop_reason="tool_use").reply(text("ok"))
    app = asking_app(tmp_path, api, timeout_s=0.2)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "pick a kata")
        await open_choice(app, pilot)
        await until(pilot, lambda: not app.generating)
        [row] = app.choices.entries()
    assert tool_result(api) == "cancel:None" and row.result.status == "timed_out" and not row.result.answers


async def test_a_question_waiting_for_the_human_is_not_counted_against_the_call_timeout(tmp_path: Path) -> None:
    api = FakeApi().reply(tool_use("mcp__fake__pick", {}), stop_reason="tool_use").reply(text("ok"))
    app = asking_app(tmp_path, api, call_timeout_s=0.6)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "pick a kata")
        await open_choice(app, pilot)
        await pilot.pause(1.4)  # well past the 0.6 s call timeout
        await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating)
    assert tool_result(api) == "accept:word-wrap"


async def test_a_form_a_choice_cannot_show_is_declined_with_a_note_and_no_widget(tmp_path: Path) -> None:
    api = FakeApi().reply(tool_use("mcp__fake__name", {}), stop_reason="tool_use").reply(text("ok"))
    app = asking_app(tmp_path, api)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "ask my name")
        await until(pilot, lambda: not app.generating)
        assert not app.choices.entries()
        shown = "\n".join(str(n.render()) for n in app.query(".note"))
    assert tool_result(api) == "decline" and "mcp fake: declined an elicitation: field 'name'" in shown


# --- the mapping, without a UI ------------------------------------------------------------------------------------


def test_a_yes_no_and_a_list_of_values_become_questions_and_read_back_typed() -> None:
    schema = {"properties": {
        "go": {"type": "boolean", "title": "Proceed"},
        "kinds": {"type": "array", "items": {"enum": ["a", "b", "c"]}}}}
    request, fields = form_request("Configure", schema)
    assert [q.multi_select for q in request.questions] == [False, True]
    assert [o.label for o in request.questions[0].options] == ["Yes", "No"]  # allowed here: a server's form is not an Approval
    assert fields[0].decode(("No",), None) is False and fields[1].decode(("a", "c"), None) == ["a", "c"]


def test_titled_values_show_their_title_and_return_their_value() -> None:
    schema = {"properties": {"c": {"oneOf": [{"const": "w", "title": "Word wrap"}, {"const": "f", "title": "FizzBuzz"}]}}}
    request, [field] = form_request("Which?", schema)
    assert [o.label for o in request.questions[0].options] == ["Word wrap", "FizzBuzz"]
    assert field.decode(("FizzBuzz",), None) == "f"


def test_forms_that_cannot_be_shown_are_unsupported() -> None:
    schemas: list[dict[str, Any]] = [
        {"properties": {}}, {"properties": {"n": {"type": "string"}}},
        {"properties": {"e": {"enum": ["1", "2", "3", "4", "5"]}}}]  # the last fails the Choice's own limit of four
    for schema in schemas:
        with pytest.raises((Unsupported, ValidationError)):
            form_request("?", schema)


async def test_the_elicitor_runs_the_asker_on_the_app_loop_and_maps_every_status() -> None:
    from harness.choice import Answer, ChoiceResult

    loop = asyncio.get_running_loop()
    seen: list[tuple[str, int]] = []
    outcomes = [
        ChoiceResult(status="answered", answers={"Which?": Answer(picked=("a",))}), ChoiceResult(status="declined"),
        ChoiceResult(status="cancelled"), ChoiceResult(status="timed_out")]

    async def asker(request, source):  # type: ignore[no-untyped-def]
        import threading
        seen.append((source, threading.get_ident()))
        return outcomes.pop(0)

    elicitor = Elicitor(asker, lambda: loop)
    params = types.ElicitRequestFormParams(
        message="Which?", requested_schema={"type": "object", "properties": {"c": {"enum": ["a", "b"]}}})
    got = [await elicitor("srv", params) for _ in range(4)]
    assert [(r.action, r.content) for r in got if isinstance(r, types.ElicitResult)] == [
        ("accept", {"c": "a"}), ("decline", None), ("cancel", None), ("cancel", None)]
    assert {s for s, _ in seen} == {"mcp:srv"}
