import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from strands.hooks import HookRegistry
from textual.pilot import Pilot
from textual.widgets import Input, Static

from harness.app import HarnessApp, Replier
from harness.chat import ContextPanel, Reply
from harness.config import HarnessConfig, load_config
from harness.hooks import CallRecorded, DecisionRecorded, GuardStateChanged, QueueChanged, TrimRecorded


async def fast_reply(prompt: str) -> AsyncIterator[str]:
    for tok in ["echo", ": ", prompt]:
        await asyncio.sleep(0)
        yield tok


async def slow_reply(prompt: str) -> AsyncIterator[str]:
    yield "start "
    await asyncio.sleep(60)
    yield "never"


def test_load_config(tmp_path: Path) -> None:
    p = tmp_path / "h.toml"
    p.write_text('[profile]\nmodel = "x"\nnum_ctx = 4096\n')
    cfg = load_config(p)
    assert (cfg.profile.model, cfg.profile.num_ctx) == ("x", 4096)
    assert load_config(tmp_path / "missing.toml") == HarnessConfig()


@pytest.mark.asyncio
async def test_ten_turn_chat_streams() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        for i in range(10):
            app.query_one(Input).value = f"msg{i}"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
        assert [r.text for r in app.query(Reply)] == [f"echo: msg{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_ctrl_c_cancels_cleanly() -> None:
    app = HarnessApp(HarnessConfig(), replier=slow_reply)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "hi"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.generating
        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
        assert not app.generating
        # the queue is paused: a new prompt waits until Enter on an empty input resumes it
        app.query_one(Input).value = "again"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert not app.generating and app.queue.pending == ("again",)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.generating


@pytest.mark.asyncio
async def test_status_bar_follows_the_watchdog_and_goes_red_on_spill() -> None:
    from tests.guard.fakes import GIB, FakeBackend
    from harness.guard.watchdog import Watchdog
    from harness.status import StatusBar

    backend = FakeBackend(models={"m": (10 * GIB, 0.0)})
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.05, memory_level=lambda: 1)
    app = HarnessApp(HarnessConfig(), replier=fast_reply, watchdog=wd)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.query_one(StatusBar).has_class("green")
        backend.gpu_bytes = 4 * GIB  # a second model grabs the GPU
        await pilot.pause(0.3)
        assert app.query_one(StatusBar).has_class("red")


def make_recorder() -> tuple[list[str], Replier]:
    ran: list[str] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        await asyncio.sleep(0.25)
        yield f"reply to {prompt}"

    return ran, replier


async def submit(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    app.query_one(Input).value = text
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_queue_runs_prompts_in_order_each_with_its_own_reply() -> None:
    ran, replier = make_recorder()
    app = HarnessApp(HarnessConfig(), replier=replier)
    async with app.run_test() as pilot:
        for p in ["one", "two", "three"]:
            await submit(app, pilot, p)
        assert app.query_one(Input).border_title  # queued count is shown
        await pilot.pause(1.2)
        assert ran == ["one", "two", "three"]
        assert [r.text for r in app.query(Reply)] == ["reply to one", "reply to two", "reply to three"]
        assert not app.query_one(Input).border_title


@pytest.mark.asyncio
async def test_queue_holds_while_a_red_alert_is_active() -> None:
    from tests.guard.fakes import GIB, FakeBackend
    from harness.guard.watchdog import Watchdog

    backend = FakeBackend(models={"m": (10 * GIB, 0.0)}, gpu_bytes=4 * GIB)  # spilled from the start
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.05, memory_level=lambda: 1)
    ran, replier = make_recorder()
    app = HarnessApp(HarnessConfig(), replier=replier, watchdog=wd)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await submit(app, pilot, "held")
        await pilot.pause(0.3)
        assert ran == [] and app.queue.pending == ("held",)
        backend.gpu_bytes = 100 * GIB  # spill clears
        await pilot.pause(0.5)
        assert ran == ["held"]


@pytest.mark.asyncio
async def test_ctrl_c_pauses_queue_then_enter_resumes_and_esc_clears() -> None:
    ran: list[str] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        await asyncio.sleep(1.5 if prompt in ("first", "slow") else 0.01)
        yield prompt

    app = HarnessApp(HarnessConfig(), replier=replier)
    async with app.run_test() as pilot:
        for p in ["first", "second", "third"]:
            await submit(app, pilot, p)
        await pilot.pause(0.05)
        await pilot.press("ctrl+c")
        await pilot.pause(0.2)
        assert ran == ["first"] and app.queue.pending == ("second", "third") and app.queue.paused
        await pilot.press("enter")  # empty input resumes; second then third follow
        await pilot.pause(0.3)
        assert ran == ["first", "second", "third"]

        await submit(app, pilot, "slow")
        await pilot.pause(0.05)
        await submit(app, pilot, "later")
        await pilot.press("ctrl+c")
        await pilot.press("escape")
        await pilot.pause(0.5)
        assert not app.queue and "later" not in ran


@pytest.mark.asyncio
async def test_up_down_recall_prompts_and_restore_draft() -> None:
    _, replier = make_recorder()
    app = HarnessApp(HarnessConfig(), replier=replier)
    async with app.run_test() as pilot:
        for p in ["a", "b", "b", ""]:
            await submit(app, pilot, p)
        await pilot.pause(0.4)
        box = app.query_one(Input)
        box.value = "draft"
        await pilot.press("up")
        assert box.value == "b"
        await pilot.press("up")
        assert box.value == "a"
        await pilot.press("down")
        await pilot.press("down")
        assert box.value == "draft"


@pytest.mark.asyncio
async def test_streamed_markdown_with_unclosed_fence_equals_one_shot_render() -> None:
    text = "# Title\n\nsome **bold** and `code` and snake_case_name, 2 * 3 * 4, #notaheading\n\n- item\n\n```py\nx = 1\n"
    chunks = [text[i : i + 3] for i in range(0, len(text), 3)]

    async def streaming(prompt: str) -> AsyncIterator[str]:
        for c in chunks:
            await asyncio.sleep(0)
            yield c

    app = HarnessApp(HarnessConfig(), replier=streaming)
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)
        streamed = app.query_one(Reply)
        oneshot = Reply()
        await app.query_one("#chat").mount(oneshot)
        await oneshot.finish(text)
        await pilot.pause(0.1)

        def shape(r: Reply) -> list[tuple[str, str]]:
            return [(type(w).__name__, str(w.render())) for w in r.walk_children() if isinstance(w, Static)]

        assert streamed.text == text and shape(streamed) == shape(oneshot)
        rendered = " ".join(s for _, s in shape(streamed))
        for literal in ("snake_case_name", "2 * 3 * 4", "#notaheading"):
            assert literal in rendered


@pytest.mark.asyncio
async def test_slash_exit_runs_immediately_while_generating_and_skips_the_queue() -> None:
    app = HarnessApp(HarnessConfig(), replier=slow_reply)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "hi"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.generating
        app.query_one(Input).value = "/exit"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.queue.pending == ()
    assert not app._running


@pytest.mark.asyncio
async def test_unknown_slash_command_is_noted_not_queued() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "/nope"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.queue.pending == () and not app.generating
        assert any("unknown command: /nope" in str(n.render()) for n in app.query(".note"))


@pytest.mark.asyncio
async def test_model_command_waits_for_queued_prompts_but_exit_does_not() -> None:
    ran, replier = make_recorder()
    app = HarnessApp(HarnessConfig(), replier=replier)
    switched: list[str] = []

    async def fake_model_command(args: str) -> None:
        switched.append(args)
        ran.append(f"/model {args}")

    app._model_command = fake_model_command  # type: ignore[method-assign,assignment]
    async with app.run_test() as pilot:
        for text in ["one", "two", "/model other", "three"]:
            await submit(app, pilot, text)
        await pilot.pause(1.5)
        assert ran == ["one", "two", "/model other", "three"]
        assert switched == ["other"]
        assert [r.text for r in app.query(Reply)] == ["reply to one", "reply to two", "reply to three"]


@pytest.mark.asyncio
async def test_model_without_backend_says_so() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/model x")
        await pilot.pause(0.3)
        assert any("no model backend" in str(n.render()) for n in app.query(".note"))


class Recorder:
    """A HookProvider (what an OpenTelemetry provider will be) that remembers every harness event."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        for event_type in (GuardStateChanged, CallRecorded, TrimRecorded, DecisionRecorded, QueueChanged):
            registry.add_callback(event_type, lambda e: self.events.append(e))

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


@pytest.mark.asyncio
async def test_components_publish_through_app_hooks_and_the_panel_and_bar_follow() -> None:
    from tests.guard.fakes import GIB, FakeBackend
    from harness.context.trimlog import TrimEntry
    from harness.decide.log import Decision
    from harness.guard.watchdog import Watchdog
    from harness.status import StatusBar

    backend = FakeBackend(models={"m": (10 * GIB, 1.0)})
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.05, memory_level=lambda: 1)
    app = HarnessApp(HarnessConfig(), replier=fast_reply, watchdog=wd)
    recorder = Recorder()
    app.hooks.add_hook(recorder)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app.ledger.before_call("m", [{"role": "user", "content": "hello there"}], None)
        app.trimlog.record(TrimEntry("summarize", "test", 2, 100, 40))
        app.decisions.record(Decision("policy", "tev1", "Action: shell", {"safe": {"type": "noul", "noul": 0.9}}))
        app.queue.put("queued")
        await pilot.pause(0.2)
        assert recorder.of(GuardStateChanged) and recorder.of(GuardStateChanged)[-1].state is wd.state
        assert recorder.of(CallRecorded)[0].record is app.ledger.pending
        assert recorder.of(TrimRecorded)[0].entry.reason == "test"
        assert recorder.of(DecisionRecorded)[0].decision.kind == "policy"
        assert recorder.of(QueueChanged)[0].pending == ("queued",)
        assert "estimate" in str(app.query_one(ContextPanel).render())  # the panel is a consumer of CallRecorded
        assert "m" in str(app.query_one(StatusBar).render())
        notes = [str(n.render()) for n in app.query(Static) if n.has_class("note")]
        assert any(n.startswith("trim ") for n in notes) and any(n.startswith("decide ") for n in notes)


def test_components_are_subscribed_once_by_the_hooks_adapter() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    app.hooks.attach_trimlog(app.trimlog)  # attaching again must not double-publish
    app.hooks.attach_ledger(app.ledger)
    assert len(app.trimlog._listeners) == 1 and len(app.ledger._listeners) == 1 and len(app.decisions._listeners) == 1
    assert len(app.queue._listeners) == 1
