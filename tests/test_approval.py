"""M8 gate: a write or shell command never runs without approval, tev1's advice notwithstanding."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from strands import Agent
from strands.hooks import BaseHookEvent
from strands.hooks.events import BeforeToolCallEvent
from strands.models import Model
from strands.types.streaming import StreamEvent
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.approval import ALWAYS_GATED, Advice, ApprovalGate, ApprovalRequest
from harness.chat import ApprovalPrompt
from harness.config import HarnessConfig, ToolsConfig
from harness.hooks import ApprovalRequested, ApprovalResolved, HarnessHooks
from harness.tools.builtin import build_shell, build_write_file


class ScriptedModel(Model):
    """Replays scripted turns: each is either a tool call (name, input) or a final text."""

    def __init__(self, *turns: tuple[str, dict[str, Any]] | str) -> None:
        self.turns = list(turns)

    def update_config(self, **model_config: Any) -> None: ...

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError
        yield

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        turn = self.turns.pop(0) if self.turns else "done"
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(turn, str):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": turn}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, args_in = turn
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args_in)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}


def agent_for(project: Path, gate: ApprovalGate | None, *turns: tuple[str, dict[str, Any]] | str) -> Agent:
    return Agent(
        model=ScriptedModel(*turns), callback_handler=None,
        tools=[build_write_file(project), build_shell(project)], hooks=[gate] if gate is not None else [],
    )


def approver(answer: bool | BaseException, asked: list[ApprovalRequest]) -> Any:
    async def ask(request: ApprovalRequest) -> bool:
        asked.append(request)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return ask


async def safe(_tool: str, _input: Mapping[str, Any]) -> float | None:
    return 0.99


async def broken(_tool: str, _input: Mapping[str, Any]) -> float | None:
    raise RuntimeError("tev1 is down")


CALLS = {
    "write_file": ("write_file", {"path": "sub/out.txt", "content": "hello"}),
    "shell": ("shell", {"command": "touch marker"}),
}
EFFECT = {"write_file": "sub/out.txt", "shell": "marker"}


@pytest.mark.parametrize("tool", ["write_file", "shell"])
async def test_denied_call_never_runs_even_when_tev1_says_safe(tool: str, tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    gate = ApprovalGate(approver(False, asked), safe)
    agent = agent_for(tmp_path, gate, CALLS[tool], "ok")
    result = await agent.invoke_async("go")
    assert not (tmp_path / EFFECT[tool]).exists()
    assert [(r.tool, r.advice.p_safe) for r in asked] == [(tool, 0.99)]
    results = [c["toolResult"] for m in agent.messages for c in m["content"] if "toolResult" in c]
    assert [(r["status"], r["content"][0]["text"]) for r in results] == [("error", f"{tool} was denied by the user")]
    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize("tool", ["write_file", "shell"])
async def test_approved_call_runs(tool: str, tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    await agent_for(tmp_path, ApprovalGate(approver(True, asked), safe), CALLS[tool], "ok").invoke_async("go")
    assert (tmp_path / EFFECT[tool]).exists() and len(asked) == 1


@pytest.mark.parametrize("tool", ["write_file", "shell"])
@pytest.mark.parametrize("answer", [False, True])
async def test_a_failing_advisor_still_asks_the_approver(tool: str, answer: bool, tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    gate = ApprovalGate(approver(answer, asked), broken)
    await agent_for(tmp_path, gate, CALLS[tool], "ok").invoke_async("go")
    assert len(asked) == 1 and asked[0].advice.p_safe is None and "tev1 is down" in asked[0].advice.error
    assert (tmp_path / EFFECT[tool]).exists() is answer


async def test_a_hanging_advisor_times_out_and_the_approver_is_still_asked(tmp_path: Path) -> None:
    async def hang(_tool: str, _input: Mapping[str, Any]) -> float | None:
        await asyncio.sleep(60)
        return 1.0

    asked: list[ApprovalRequest] = []
    gate = ApprovalGate(approver(False, asked), hang, advice_timeout=0.05)
    await agent_for(tmp_path, gate, CALLS["shell"], "ok").invoke_async("go")
    assert len(asked) == 1 and asked[0].advice.p_safe is None


@pytest.mark.parametrize("tool", ["write_file", "shell"])
async def test_an_approver_that_raises_cancels_the_tool(tool: str, tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    gate = ApprovalGate(approver(RuntimeError("ui crashed"), asked), safe)
    await agent_for(tmp_path, gate, CALLS[tool], "ok").invoke_async("go")
    assert len(asked) == 1 and not (tmp_path / EFFECT[tool]).exists()


async def test_a_non_bool_answer_is_not_an_approval(tmp_path: Path) -> None:
    async def maybe(_request: ApprovalRequest) -> bool:
        return cast(bool, "yes")

    await agent_for(tmp_path, ApprovalGate(maybe, safe), CALLS["shell"], "ok").invoke_async("go")
    assert not (tmp_path / "marker").exists()


async def test_ungated_tools_skip_the_approver(tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    gate = ApprovalGate(approver(False, asked), safe)
    event = BeforeToolCallEvent(agent=cast(Any, None), selected_tool=None,
                                tool_use={"toolUseId": "1", "name": "read_file", "input": {}}, invocation_state={})
    await gate.before_tool(event)
    assert asked == [] and event.cancel_tool is False


def test_shell_and_write_file_are_always_gated() -> None:
    assert ToolsConfig(approve=[]).approve == ["shell", "write_file"]
    assert ToolsConfig(approve=["extra"]).approve == ["extra", "shell", "write_file"]
    assert ApprovalGate(approver(True, []), gated=[]).gated == ALWAYS_GATED
    assert build_shell(Path(".")).tool_name in ALWAYS_GATED


async def test_gate_publishes_request_and_resolution_through_hooks(tmp_path: Path) -> None:
    hooks = HarnessHooks()
    seen: list[BaseHookEvent] = []
    hooks.add_callback(ApprovalRequested, lambda e: seen.append(e))
    hooks.add_callback(ApprovalResolved, lambda e: seen.append(e))
    gate = ApprovalGate(approver(True, []), safe, hooks=hooks)
    await agent_for(tmp_path, gate, CALLS["shell"], "ok").invoke_async("go")
    assert [type(e) for e in seen] == [ApprovalRequested, ApprovalResolved]
    resolved = seen[1]
    assert isinstance(resolved, ApprovalResolved) and resolved.approved and resolved.request.label() == "shell: touch marker"


def test_advice_lines() -> None:
    assert Advice(0.91).line() == "tev1: safe p=0.91 (advisory)"
    assert Advice(0.2).line() == "tev1: risky p=0.20 (advisory)"
    assert Advice().line() == "tev1: unavailable"


# --- app level: the approval prompt and the Prompt queue ------------------------------------------


def make_app(ran: list[str], answers: list[bool]) -> HarnessApp:
    holder: list[HarnessApp] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        ok = await holder[0].ask_approval(ApprovalRequest("shell", {"command": f"echo {prompt}"}, Advice(0.99)))
        answers.append(ok)
        yield f"{prompt}: {'approved' if ok else 'denied'}"

    app = HarnessApp(HarnessConfig(), replier=replier)
    holder.append(app)
    return app


def records(app: HarnessApp) -> list[str]:
    """The one-line records answered inline prompts leave in the chat log."""
    return [str(n.render()) for n in app.query(Static) if n.has_class("record")]


async def submit(app: HarnessApp, text: str) -> None:
    box = app.query_one(Input)
    box.value = text
    await box.action_submit()


async def test_queued_prompts_wait_for_the_pending_approval_then_run_in_order() -> None:
    ran: list[str] = []
    answers: list[bool] = []
    app = make_app(ran, answers)
    async with app.run_test() as pilot:
        await submit(app, "one")
        await pilot.pause(0.1)
        assert app.approval_pending and ran == ["one"]
        prompt = app.query_one(ApprovalPrompt)  # mounted in the chat log when asked, not part of the layout
        assert prompt.parent is app.query_one("#chat") and app.focused is prompt
        await submit(app, "two")
        await submit(app, "three")
        await pilot.pause(0.2)
        assert ran == ["one"] and app.queue.pending == ("two", "three")
        assert app.query_one(Input).border_title == "2 queued · held by approval"
        for expected in (["one"], ["one", "two"], ["one", "two", "three"]):
            await pilot.press("y")
            await pilot.pause(0.2)
            assert ran[: len(expected)] == expected
        await pilot.press("n")  # nothing pending any more: a stray key does nothing
        await app.workers.wait_for_complete()
        assert answers == [True, True, True] and not app.approval_pending
        assert not app.query(ApprovalPrompt) and app.focused is app.query_one(Input)
        assert records(app) == [f"✓ shell: echo {p}? approved" for p in ("one", "two", "three")]


async def test_pump_refuses_to_start_while_an_approval_is_pending_even_when_idle() -> None:
    ran: list[str] = []
    app = make_app(ran, [])
    async with app.run_test() as pilot:
        waiting = asyncio.create_task(app.ask_approval(ApprovalRequest("write_file", {"path": "x", "content": ""})))
        await pilot.pause(0.1)
        assert app.approval_pending and not app.generating
        await submit(app, "held")
        app._pump()
        await pilot.pause(0.2)
        assert ran == [] and app.queue.pending == ("held",)
        await pilot.press("n")
        assert await waiting is False
        await pilot.pause(0.2)
        assert ran == ["held"]  # resolving the approval released the queue
        await pilot.press("n")
        await app.workers.wait_for_complete()


async def test_escape_denies() -> None:
    app = make_app([], [])
    async with app.run_test() as pilot:
        waiting = asyncio.create_task(app.ask_approval(ApprovalRequest("shell", {"command": "ls"})))
        await pilot.pause(0.1)
        await pilot.press("escape")
        assert await waiting is False


async def test_ctrl_c_during_an_approval_denies_it_and_pauses_the_queue() -> None:
    ran: list[str] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        await asyncio.sleep(60)
        yield "never"

    app = HarnessApp(HarnessConfig(), replier=replier)
    async with app.run_test() as pilot:
        await submit(app, "one")
        await pilot.pause(0.1)
        waiting = asyncio.create_task(app.ask_approval(ApprovalRequest("shell", {"command": "ls"})))
        await submit(app, "two")
        await pilot.pause(0.1)
        assert app.approval_pending and app.generating
        await pilot.press("ctrl+c")
        assert await asyncio.wait_for(waiting, 1) is False  # denied, not left hanging
        await pilot.pause(0.2)
        assert not app.approval_pending and not app.generating
        assert app.queue.paused and app.queue.pending == ("two",) and ran == ["one"]
        assert not app.query(ApprovalPrompt) and records(app) == ["✗ shell: ls? denied"]


async def test_the_gate_drives_the_prompt_and_notes_the_decision_in_the_chat(tmp_path: Path) -> None:
    app = HarnessApp(HarnessConfig(), replier=lambda p: _empty())
    gate = ApprovalGate(app.ask_approval, safe, hooks=app.hooks)
    async with app.run_test() as pilot:
        event = BeforeToolCallEvent(agent=cast(Any, None), selected_tool=None,
                                    tool_use={"toolUseId": "1", "name": "shell", "input": {"command": "ls"}}, invocation_state={})
        task = asyncio.create_task(gate.before_tool(event))
        await pilot.pause(0.1)
        shown = str(app.query_one(ApprovalPrompt).render())
        assert "command: ls" in shown and "tev1: safe p=0.99 (advisory)" in shown
        assert event.cancel_tool  # closed until the user says yes
        await pilot.press("y")
        await task
        assert not event.cancel_tool
        await pilot.pause(0.1)
        assert records(app) == ["✓ shell: ls? approved"]
        task = asyncio.create_task(gate.before_tool(event))
        await pilot.pause(0.1)
        await pilot.press("n")
        await task
        assert event.cancel_tool
        await pilot.pause(0.1)
        assert records(app) == ["✓ shell: ls? approved", "✗ shell: ls? denied"]


async def _empty() -> AsyncIterator[str]:
    return
    yield
