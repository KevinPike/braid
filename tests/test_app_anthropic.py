"""F1b gates in the TUI: the chat agent on the Anthropic API, with no Ollama request, usage in the status bar, compaction
that strips thinking blocks, and Ctrl-C that closes the stream. The Messages API is faked (``tests.anthropic_fakes``)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Input

from harness.app import HarnessApp
from harness.chat import Reply
from harness.config import ContextConfig, DecideConfig, HarnessConfig, PathsConfig, ProfileConfig, ToolsConfig
from harness.status import StatusBar
from tests.anthropic_fakes import MODEL, FakeApi, text, thinking, tool_use


@pytest.fixture(autouse=True)
def no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate 3: any Ollama-facing object built, or any guard call made, fails the test."""
    def fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an Ollama-facing call was made on the anthropic backend")

    import harness.app as app_module

    for name in ("OllamaClient", "Watchdog", "TevClient", "run_preflight", "make_summarizer", "system_gpu_limit"):
        monkeypatch.setattr(app_module, name, fail)


def config(tmp_path: Path, **profile: Any) -> HarnessConfig:
    tools_root = tmp_path / "project"
    tools_root.mkdir(exist_ok=True)
    (tools_root / "hello.txt").write_text("hi from the file\n")
    return HarnessConfig(
        profile=ProfileConfig(**profile), paths=PathsConfig(data_dir=tmp_path / "data"),
        tools=ToolsConfig(root=tools_root, shell=False, write=False),
        context=ContextConfig(preserve_recent=2, offload_tokens=0), decide=DecideConfig(enabled=True),
    )


async def until(pilot: Pilot[None], predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for the condition")
        await pilot.pause(0.02)


async def send(app: HarnessApp, pilot: Pilot[None], prompt: str) -> None:
    app.query_one("#prompt", Input).value = prompt
    await pilot.press("enter")
    await pilot.pause(0.05)
    await until(pilot, lambda: not app.generating)


def status(app: HarnessApp) -> str:
    return str(app.query_one(StatusBar).render())


async def test_a_text_reply_and_a_tool_round_trip_reach_the_chat_with_no_ollama_request(tmp_path: Path) -> None:
    api = (FakeApi()
           .reply(thinking(), text("Hello there"))
           .reply(thinking("sig-2"), tool_use("read_file", {"path": "hello.txt"}), stop_reason="tool_use")
           .reply(text("The file says hi.")))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        assert app._client is None and app.tev is None and app.guard_state is None  # no Guard on this backend
        await send(app, pilot, "hi")
        await send(app, pilot, "what is in hello.txt?")
        replies = [r.text for r in app.query(Reply)]
        assert replies[0] == "Hello there" and replies[1].strip().endswith("The file says hi.")
        result = api.bodies[2]["messages"][-1]["content"][0]
        assert result["type"] == "tool_result" and "hi from the file" in result["content"][0]["text"]
        assert len(app.store.calls(app.session_id)) == 3  # every API request is in the session's calls table
        assert app.store.calls(app.session_id)[0].prompt_eval_count == 1120


async def test_the_status_bar_shows_tokens_and_cost_from_usage(tmp_path: Path) -> None:
    usage = {"input_tokens": 2000, "output_tokens": 400, "cache_read_input_tokens": 10_000, "cache_creation_input_tokens": 0}
    api = FakeApi().reply(text("one"), usage=usage).reply(text("two"), usage=usage)
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        assert status(app).startswith(MODEL) and "$0.0000" in status(app)
        await send(app, pilot, "a")
        await send(app, pilot, "b")
        bar = status(app)
        cost = 2 * (2000 * 0.10 + 10_000 * 0.01 + 400 * 0.50) / 1e6
        assert f"${cost:.4f}" in bar
        assert "ctx 12.4k/1M (1%)" in bar  # 2000 + 10000 prompt tokens + 400 out, against the real window
        assert "in 24k · out 800 · cache 20k" in bar
        assert "GPU" not in bar and "loading" not in bar


async def test_compaction_against_the_window_strips_thinking_from_the_kept_turns(tmp_path: Path) -> None:
    big = {"input_tokens": 1500, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    api = (FakeApi()
           .reply(thinking("sig-1"), text("first answer"))
           .reply(thinking("sig-2", "a summarized thought (F9)"), text("second answer"), usage=big)  # 1600 of a 2000 window: over the 90% ceiling
           .reply(text("- the user asked two things"))  # the summarizer's (non-streamed) call
           .reply(text("third answer")))
    app = HarnessApp(config(tmp_path, context_window=2000), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "q1")
        await send(app, pilot, "q2")
        assert len(api.bodies) == 3 and "stream" not in api.bodies[2]  # compaction ran after the second turn
        assert "q1" in api.bodies[2]["messages"][0]["content"]
        await send(app, pilot, "q3")
        body = api.bodies[3]
        blocks = [b for m in body["messages"] for b in m["content"]]
        assert not [b for b in blocks if b["type"] in ("thinking", "redacted_thinking")]  # the strategy: strip them
        assert body["messages"][0]["content"][0]["text"].startswith("[Summary of the earlier conversation]")
        assert {"type": "text", "text": "second answer"} in blocks  # the kept turn is still there, without its thinking
        assert app.trimlog.entries()[-1].detail.endswith("stripped 1 thinking blocks")
        assert app.meter is not None and app.meter.state.calls == 4  # the summarizer's call is priced too


async def test_ctrl_c_closes_the_stream_and_the_next_turn_starts_from_a_consistent_history(tmp_path: Path) -> None:
    api = FakeApi(hang_after=4).reply(text("a long answer that never ends")).reply(text("after"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        app.query_one("#prompt", Input).value = "hi"
        await pilot.press("enter")
        await asyncio.wait_for(api.started.wait(), 5)
        await pilot.pause(0.05)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: api.closed == 1)
        assert not app.generating and app.meter is not None and not app.meter.state.in_flight
        api.hang_after = None
        await pilot.press("enter")  # resume the paused queue (empty)
        await send(app, pilot, "again")
        assert [r.text for r in app.query(Reply)][-1] == "after"
        # the cancelled turn is cut back like a resume would: no half reply, and no unanswered prompt
        assert api.bodies[1]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "again"}]}]


async def test_a_resumed_api_session_is_adopted_once_and_continues_on_its_model(tmp_path: Path) -> None:
    from harness.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    old = store.create(MODEL, 1_000_000)
    store.save_messages(old.id, [
        {"role": "user", "content": [{"text": "earlier"}]},
        {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "", "signature": "s"}}}, {"text": "earlier answer"}]},
    ])
    api = FakeApi().reply(text("next"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client(), store=store, resume=old.id)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None and len(app._messages) == 2)
        await pilot.pause(0.1)
        notes = [str(w.render()) for w in app.query(".note")]
        assert sum(n.startswith(f"session {old.id}") for n in notes) == 1  # adopted exactly once
        await send(app, pilot, "go on")
        sent = api.bodies[0]["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]
        assert sent[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "s"}  # unedited history keeps it
