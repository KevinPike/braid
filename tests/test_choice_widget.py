"""C2 gates: the Choice widget in the chat log, its keys, and its interaction with the Prompt queue, the human lock
(Approvals) and Ctrl-C (ADR 0010). A fake replier asks the Choice; the Textual pilot answers it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.approval import ApprovalRequest
from harness.chat import ApprovalPrompt, ChoicePrompt, NoteInput, OtherInput
from harness.choice import ChoiceRecord, ChoiceRequest, ChoiceResult, validate
from harness.config import HarnessConfig
from harness.hooks import ChoiceRequested, ChoiceResolved


def request(*questions: dict[str, Any]) -> ChoiceRequest:
    return validate({"questions": list(questions)})


STORAGE: dict[str, Any] = {
    "header": "Storage", "question": "Where should the kata keep its users?", "options": [
        {"label": "SQLite", "description": "one file", "preview": "CREATE TABLE users (id);"},
        {"label": "JSON file", "description": "simplest"}, {"label": "In memory"}]}
HASH: dict[str, Any] = {
    "header": "Hashing", "question": "Which hash?", "multiSelect": True,
    "options": [{"label": "argon2"}, {"label": "bcrypt"}, {"label": "scrypt"}]}


class Rig:
    """An app whose replier asks ``self.request`` and keeps what came back."""

    def __init__(self, req: ChoiceRequest | None = None) -> None:
        self.request = req or request(STORAGE)
        self.results: list[ChoiceResult] = []
        self.ran: list[str] = []
        self.app = HarnessApp(HarnessConfig(), replier=self.replier)

    async def replier(self, prompt: str) -> AsyncIterator[str]:
        self.ran.append(prompt)
        self.results.append(await self.app.ask_choice(self.request))
        yield f"{prompt}: {self.results[-1].status}"


async def until(pilot: Pilot[None], predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for the condition")
        await pilot.pause(0.02)


async def submit(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")


async def open_choice(rig: Rig, pilot: Pilot[None], prompt: str = "go") -> ChoicePrompt:
    await submit(rig.app, pilot, prompt)
    await until(pilot, lambda: bool(rig.app.query(ChoicePrompt)))
    await pilot.pause(0.05)
    return rig.app.query_one(ChoicePrompt)


def records(app: HarnessApp) -> list[str]:
    return [str(n.render()) for n in app.query(Static) if n.has_class("record")]


def text_of(app: HarnessApp, widget_id: str) -> str:
    return str(app.query_one(ChoicePrompt).query_one(widget_id, Static).render())


async def test_pressing_2_then_enter_returns_that_label_and_the_record_replaces_the_widget() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        widget = await open_choice(rig, pilot)
        assert widget.has_focus and rig.app.focused is widget
        await pilot.press("2", "enter")
        await until(pilot, lambda: bool(rig.results))
        await until(pilot, lambda: not rig.app.query(ChoicePrompt) and bool(records(rig.app)))
        assert rig.results[0].for_model() == {"answers": {"Where should the kata keep its users?": "JSON file"}}
        assert records(rig.app)[0].startswith("✓ Storage? JSON file (") and records(rig.app)[0].endswith(" s)")
        await until(pilot, lambda: rig.app.focused is not None and rig.app.focused.id == "prompt")


async def test_enter_alone_takes_the_highlighted_option_and_arrows_move() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press("down", "down", "up", "enter")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].answers[STORAGE["question"]].picked == ("JSON file",)


async def test_space_toggles_in_a_multi_select_and_enter_needs_a_pick() -> None:
    rig = Rig(request(HASH))
    async with rig.app.run_test() as pilot:
        widget = await open_choice(rig, pilot)
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert not rig.results and "pick at least one" in text_of(rig.app, "#hint") and widget.is_open
        await pilot.press("space", "down", "down", "space", "space", "enter")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].answers["Which hash?"].picked == ("argon2",)  # scrypt was ticked and unticked


async def test_multi_select_keeps_the_ticked_set() -> None:
    rig = Rig(request(HASH))
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press("1", "3", "enter")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].for_model() == {"answers": {"Which hash?": "argon2, scrypt"}}


async def test_other_with_typed_text_returns_other_and_empty_text_is_rejected() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        widget = await open_choice(rig, pilot)
        await pilot.press("4")
        await pilot.pause(0.05)
        assert isinstance(rig.app.focused, OtherInput)
        await pilot.press("enter")  # empty
        await pilot.pause(0.05)
        assert not rig.results and "Other needs some text" in text_of(rig.app, "#hint") and widget.is_open
        await pilot.press(*"redis")
        await pilot.press("enter")
        await until(pilot, lambda: bool(rig.results))
        answer = rig.results[0].answers[STORAGE["question"]]
        assert (answer.picked, answer.other) == (("Other",), "redis")
        assert rig.results[0].for_model() == {"answers": {STORAGE["question"]: "redis"}}


async def test_n_adds_a_note() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press("1", "n")
        await pilot.pause(0.05)
        assert isinstance(rig.app.focused, NoteInput)
        await pilot.press(*"fast", "enter")  # Enter in the note returns to the list
        await pilot.pause(0.05)
        assert not rig.results and rig.app.query_one(ChoicePrompt).has_focus
        await pilot.press("enter")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].for_model() == {
            "answers": {STORAGE["question"]: "SQLite"}, "notes": {STORAGE["question"]: "fast"}}


@pytest.mark.parametrize("keys", [(), ("4",), ("n",)])  # the list, the Other field and the note field
async def test_esc_declines_even_while_an_input_has_focus(keys: tuple[str, ...]) -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press(*keys)
        await pilot.pause(0.05)
        await pilot.press("escape")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].for_model() == {"status": "declined", "reason": "the user declined to answer"}
        await until(pilot, lambda: "✗ choice declined" in records(rig.app))


async def test_the_preview_pane_shows_only_for_an_option_with_a_preview() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        pane = rig.app.query_one(ChoicePrompt).query_one("#preview", Static)
        assert pane.display and "CREATE TABLE users" in str(pane.render())
        await pilot.press("down")
        await pilot.pause(0.05)
        assert not pane.display
        await pilot.press("up")
        await pilot.pause(0.05)
        assert pane.display


async def test_model_text_is_not_parsed_as_markup() -> None:
    rig = Rig(request({**STORAGE, "question": "Use [bold]this[/bold] or [red]that[/red]?"}))
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        assert "[bold]this[/bold]" in text_of(rig.app, "#ask")


async def test_a_secret_in_the_request_is_shown_redacted_but_answered_with_the_real_label() -> None:
    key = "sk-ant-api03-" + "aB3dE5fG7hJ9kL1mN3pQ5rS7"
    rig = Rig(request({**STORAGE, "options": [{"label": "A", "description": key}, {"label": "B"}]}))
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        assert key not in text_of(rig.app, "#options") and "[redacted]" in text_of(rig.app, "#options")


async def test_two_questions_chips_tab_and_a_single_submit() -> None:
    rig = Rig(request(STORAGE, HASH))
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        assert "[Storage]" in text_of(rig.app, "#head") and "question 1/2" in text_of(rig.app, "#head")
        await pilot.press("3", "enter")  # answers the first and moves to the second
        await pilot.pause(0.05)
        assert "Storage ✓" in text_of(rig.app, "#head") and "question 2/2" in text_of(rig.app, "#head")
        await pilot.press("shift+tab", "tab")  # back and forth keeps the answer
        await pilot.pause(0.05)
        assert "Storage ✓" in text_of(rig.app, "#head") and not rig.results
        await pilot.press("2", "enter")
        await until(pilot, lambda: bool(rig.results))
        assert rig.results[0].for_model() == {"answers": {STORAGE["question"]: "In memory", "Which hash?": "bcrypt"}}
        await until(pilot, lambda: bool(records(rig.app)))
        assert records(rig.app)[0].startswith("✓ Storage? In memory · Hashing? bcrypt (")


async def test_a_queued_prompt_waits_for_the_choice_and_the_label_says_why() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot, "first")
        box = rig.app.query_one("#prompt", Input)  # the Choice has the focus, so Enter would answer it: submit directly
        box.value = "second"
        await box.action_submit()
        await pilot.pause(0.05)
        assert rig.ran == ["first"] and rig.app.queue.pending == ("second",)
        assert rig.app.query_one("#prompt", Input).border_title == "1 queued · held by a choice"
        await pilot.press("1", "enter")
        await until(pilot, lambda: rig.ran == ["first", "second"])  # the second starts only after the answer
        await until(pilot, lambda: bool(rig.app.query(ChoicePrompt)))
        await pilot.press("escape")
        await until(pilot, lambda: len(rig.results) == 2)


@pytest.mark.parametrize("keys", [(), ("4",), ("n",)])  # the list, the Other field and the note field must not swallow Ctrl-C
async def test_ctrl_c_with_a_choice_open_cancels_it_and_answers_nothing(keys: tuple[str, ...]) -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press(*keys)
        await pilot.pause(0.05)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not rig.app.query(ChoicePrompt) and not rig.app.generating)
        assert rig.app.queue.paused
        assert not any(r.status == "answered" for r in rig.results)
        assert "✗ choice cancelled" in records(rig.app)


async def test_a_choice_asked_after_ctrl_c_is_cancelled_without_mounting_anything() -> None:
    rig = Rig()
    async with rig.app.run_test() as pilot:
        await pilot.pause(0.05)
        result = await rig.app.ask_choice(rig.request)  # nothing is generating: the turn is over
        assert result.status == "cancelled" and not rig.app.query(ChoicePrompt)


async def test_a_choice_and_an_approval_open_one_at_a_time_in_call_order() -> None:
    rig = Rig()
    app = rig.app
    approvals: list[bool] = []

    async def both(prompt: str) -> AsyncIterator[str]:
        asked = asyncio.create_task(app.ask_choice(rig.request))
        await asyncio.sleep(0)
        approved = asyncio.create_task(app.ask_approval(ApprovalRequest("shell", {"command": "true"})))
        rig.results.append(await asked)
        approvals.append(await approved)
        yield "done"

    app._replier = both
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.pause(0.1)
        assert not app.query(ApprovalPrompt)  # the Approval waits behind the Choice
        await pilot.press("y")  # y is the Approval's key: the Choice does not read it
        await pilot.pause(0.05)
        assert app.query(ChoicePrompt) and not rig.results
        await pilot.press("1", "enter")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        assert not app.query(ChoicePrompt)
        await pilot.press("n")  # and an Approval's n is not a Choice answer
        await until(pilot, lambda: bool(approvals))
        assert approvals == [False] and rig.results[0].status == "answered"


async def test_hook_events_and_the_log_see_every_choice() -> None:
    rig = Rig()
    seen: list[Any] = []
    rig.app.hooks.add_callback(ChoiceRequested, lambda e: seen.append(e))
    rig.app.hooks.add_callback(ChoiceResolved, lambda e: seen.append(e))
    ticks = iter([10.0, 12.5])
    rig.app._clock = lambda: next(ticks)
    async with rig.app.run_test() as pilot:
        await open_choice(rig, pilot)
        await pilot.press("1", "enter")
        await until(pilot, lambda: len(seen) == 2)
        requested, resolved = seen
        assert (requested.source, requested.model) == ("chat", "gemma4:e4b")
        assert isinstance(resolved.record, ChoiceRecord) and resolved.record.latency_s == 2.5 and resolved.record.source == "chat"
        [row] = rig.app.choices.entries()
        assert (row.result.status, row.latency_s) == ("answered", 2.5)


async def test_the_preview_stacks_below_the_list_on_a_narrow_terminal() -> None:
    for size, narrow in (((120, 40), False), ((80, 30), True)):
        rig = Rig()
        async with rig.app.run_test(size=size) as pilot:
            widget = await open_choice(rig, pilot)
            await pilot.pause(0.05)
            assert widget.has_class("narrow") is narrow
