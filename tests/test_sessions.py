import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest
import respx
from strands.types.content import Message
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Static

from harness.app import HarnessApp
from harness.chat import Reply
from harness.config import HarnessConfig
from harness.context.accounting import TokenEstimator
from harness.context.ledger import Ledger
from harness.guard.ollama_client import CallMetrics, OllamaClient
from harness.inspector import Inspector
from harness.prompts import PromptHistory, PromptQueue
from harness.session.store import SessionStore, settle

NUM_CTX = 100_000  # far above any prompt here, so the ledger always learns


def msg(role: str, body: str) -> Message:
    return {"role": role, "content": [{"text": body}]}  # type: ignore[typeddict-item]


# --- the store ----------------------------------------------------------------------------------


def test_messages_snapshot_is_replaced_not_appended() -> None:
    store = SessionStore()
    s = store.create("m", 4096)
    history = [msg("user", "a"), msg("assistant", "b"), msg("user", "c")]
    store.save_messages(s.id, history)
    history[:] = [msg("user", "summary of a-c")]  # what Compaction does
    store.save_messages(s.id, history)
    assert store.messages(s.id) == history
    assert store.get(s.id) is not None and store.get(s.id).message_count == 1  # type: ignore[union-attr]


def test_prompts_global_recall_versus_per_session() -> None:
    store = SessionStore()
    a, b = store.create("m", 1), store.create("m", 1)
    for sid, text in [(a.id, "one"), (b.id, "two"), (a.id, "three")]:
        store.add_prompt(sid, text)
    assert [p.text for p in store.prompts()] == ["one", "two", "three"]
    assert [p.text for p in store.prompts(a.id)] == ["one", "three"]
    assert store.get(a.id).title == "one"  # type: ignore[union-attr]


def test_title_is_the_first_real_prompt_clipped() -> None:
    store = SessionStore()
    s = store.create("m", 1)
    store.add_prompt(s.id, "/sessions")
    store.add_prompt(s.id, "word " * 40)
    title = store.get(s.id).title  # type: ignore[union-attr]
    assert title.startswith("word word") and len(title) <= 60 and title.endswith("…")


def test_queue_is_replaced_on_every_save() -> None:
    store = SessionStore()
    s = store.create("m", 1)
    store.save_queue(s.id, ["x", "y", "z"])
    store.save_queue(s.id, ["y", "z"])
    assert store.queue(s.id) == ["y", "z"]
    store.save_queue(s.id, [])
    assert store.queue(s.id) == []


def test_calls_keep_the_exact_payload_and_are_completed_later() -> None:
    store = SessionStore()
    s = store.create("m", 1)
    body = {"model": "m", "messages": [{"role": "user", "content": "héllo"}], "options": {"num_ctx": 4096}}
    n = store.record_call(s.id, "m", 4096, body, estimate=100)
    [open_call] = store.calls(s.id)
    assert open_call.payload == body and open_call.prompt_eval_count is None and open_call.error is None
    store.complete_call(s.id, n, 110, 7)
    [done] = store.calls(s.id)
    assert (done.prompt_eval_count, done.eval_count, done.estimate) == (110, 7, 100)
    assert done.error == pytest.approx(10 / 110)
    assert store.record_call(s.id, "other", 1, {}) == 2  # numbered within the session; estimate-less is allowed


def test_fork_copies_history_prompts_and_calibration_but_not_calls_or_queue() -> None:
    store = SessionStore()
    src = store.create("m", 4096, {"estimator": {"chars": 10}})
    store.save_messages(src.id, [msg("user", "a"), msg("assistant", "b")])
    store.add_prompt(src.id, "a")
    store.save_queue(src.id, ["later"])
    store.record_call(src.id, "m", 4096, {"x": 1}, 5)
    fork = store.fork(src.id)
    assert fork.id != src.id and fork.parent_id == src.id
    assert (fork.model, fork.num_ctx, fork.calibration) == ("m", 4096, {"estimator": {"chars": 10}})
    assert store.messages(fork.id) == store.messages(src.id)
    assert [p.text for p in store.prompts(fork.id)] == ["a"]
    assert store.queue(fork.id) == [] and store.calls(fork.id) == []
    assert [p.text for p in store.prompts()] == ["a"]  # the copy does not repeat in the global recall
    # the two now evolve independently
    store.save_messages(fork.id, [msg("user", "z")])
    assert len(store.messages(src.id)) == 2


def test_resolve_by_unambiguous_prefix_and_listing_hides_empty_sessions() -> None:
    store = SessionStore()
    a, empty = store.create("m", 1), store.create("m", 1)
    store.save_messages(a.id, [msg("user", "a"), msg("assistant", "b")])
    assert store.resolve(a.id[:3]) == store.get(a.id)
    assert store.resolve("zz") is None
    assert [s.id for s in store.sessions()] == [a.id] and store.latest().id == a.id  # type: ignore[union-attr]
    assert {s.id for s in store.sessions(include_empty=True)} == {a.id, empty.id}


def test_settle_drops_an_unfinished_tail() -> None:
    use: Message = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t", "name": "r", "input": {}}}]}
    result: Message = {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "status": "success", "content": []}}]}
    done = [msg("user", "q"), msg("assistant", "a")]
    assert settle(done) == (done, 0)
    assert settle([*done, msg("user", "unanswered")]) == (done, 1)
    assert settle([*done, msg("user", "q2"), use, result]) == (done, 3)
    assert settle([]) == ([], 0)


def test_prompt_history_seeds_from_entries_and_reports_what_it_recorded() -> None:
    h = PromptHistory(["a", "a", "b"])
    assert h.previous("") == "b" and h.previous("") == "a" and h.previous("") is None
    assert h.push("c") and not h.push("c") and not h.push("  ")


def test_queue_notifies_and_restores_paused_ahead_of_new_prompts() -> None:
    seen: list[tuple[str, ...]] = []
    q = PromptQueue()
    q.subscribe(seen.append)
    q.put("new")
    q.restore(["old1", "old2"])
    assert q.pending == ("old1", "old2", "new") and q.paused and q.pop() is None
    q.resume()
    assert q.pop() == "old1"
    q.clear()
    assert seen == [("new",), ("old1", "old2", "new"), ("old2", "new"), ()]


# --- the client hook ----------------------------------------------------------------------------


@respx.mock
async def test_client_reports_the_exact_body_before_posting() -> None:
    client = OllamaClient("http://ollama.test")
    client.pin("m", 8192)
    seen: list[tuple[str, dict[str, object]]] = []
    route = respx.post("http://ollama.test/api/chat").mock(return_value=httpx.Response(
        200, content=json.dumps({"done": True, "message": {"content": "x"}, "prompt_eval_count": 3}).encode()))
    client.on_request(lambda model, body: seen.append((model, json.loads(json.dumps(body)))))
    _ = [c async for c in client.chat("m", [{"role": "user", "content": "yo"}], options={"temperature": 0.1})]
    [(model, body)] = seen
    assert model == "m" and body == json.loads(route.calls[0].request.content)
    assert body["options"] == {"temperature": 0.1, "num_ctx": 8192}


# --- the validation gate ------------------------------------------------------------------------


def ollama_tokens(messages: Sequence[dict[str, str]]) -> int:
    """What the fake Ollama measures: 4 chars per token plus 4 tokens of framing per message."""
    return sum(len(m["content"]) // 4 + 4 for m in messages)


def wire(history: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": "system", "content": "You are a concise assistant."},
            *({"role": m["role"], "content": m["content"][0]["text"]} for m in history)]


class Run:
    """A chat against the fake Ollama, persisted to a session store after every call (as the app does)."""

    def __init__(self, store: SessionStore, session_id: str, history: list[Message], estimator: TokenEstimator) -> None:
        self.store, self.session_id, self.history = store, session_id, history
        self.estimator = estimator
        self.ledger = Ledger(lambda _m: NUM_CTX, estimator)

    def call(self, prompt: str, reply: str) -> tuple[int, int]:
        """One turn; returns (harness estimate, fake-Ollama prompt_eval_count) for its request."""
        self.history.append(msg("user", prompt))
        request = wire(self.history)
        estimate = self.ledger.before_call("m", request, None)
        measured = ollama_tokens(request)
        self.ledger.after_call(CallMetrics(measured, len(reply) // 4, 1.0, 0.0))
        self.history.append(msg("assistant", reply))
        self.store.save_messages(self.session_id, self.history)
        self.store.update(self.session_id, calibration={"estimator": self.estimator.state()})
        return estimate, measured


def chatter(i: int) -> tuple[str, str]:
    return (f"question {i}: " + "why does the moon wobble so? " * (3 + i % 4),
            f"answer {i}: " + "because of tides and torque, mostly. " * (4 + i % 3))


def test_gate_resume_after_a_kill_estimates_the_next_call_within_5_percent(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    session = store.create("m", 8192)
    run = Run(store, session.id, [], TokenEstimator())
    errors = []
    for i in range(8):
        est, got = run.call(*chatter(i))
        errors.append(abs(est - got) / got)
    assert errors[0] > 0.05 > errors[-1]  # it did learn the 4.0 ratio over the 3.5 it started with
    del run, store  # kill: no close, no shutdown path

    store2 = SessionStore(db)
    info = store2.get(session.id)
    assert info is not None
    resumed = Run(store2, info.id, settle(store2.messages(info.id))[0], TokenEstimator())
    resumed.estimator.restore(info.estimator_state)
    est, got = resumed.call(*chatter(8))
    assert abs(est - got) / got < 0.05

    # without the restored estimator the same call is measurably worse: the persistence is what matters
    cold = Run(store2, info.id, settle(store2.messages(info.id))[0][:-2], TokenEstimator())
    cold_est, cold_got = cold.call(*chatter(8))
    assert abs(cold_est - cold_got) / cold_got > 0.05


# --- the app ------------------------------------------------------------------------------------


async def fast_reply(prompt: str) -> AsyncIterator[str]:
    await asyncio.sleep(0)
    yield f"echo {prompt}"


async def slow_reply(prompt: str) -> AsyncIterator[str]:
    yield "start "
    await asyncio.sleep(60)
    yield "never"


async def submit(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    app.query_one(Input).value = text
    await pilot.press("enter")


def shown(app: HarnessApp) -> str:
    return "\n".join(str(w.render()) for w in app.query(Static))


async def test_prompt_history_survives_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    app = HarnessApp(HarnessConfig(), replier=fast_reply, store=SessionStore(db))
    async with app.run_test() as pilot:
        for p in ["alpha", "beta"]:
            await submit(app, pilot, p)
        await app.workers.wait_for_complete()
    app2 = HarnessApp(HarnessConfig(), replier=fast_reply, store=SessionStore(db))  # a new session, same history
    async with app2.run_test() as pilot:
        box = app2.query_one(Input)
        await pilot.press("up")
        assert box.value == "beta"
        await pilot.press("up")
        assert box.value == "alpha"


async def test_queued_prompts_come_back_paused_and_run_only_on_enter(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ran: list[str] = []

    async def replier(prompt: str) -> AsyncIterator[str]:
        ran.append(prompt)
        await asyncio.sleep(5 if prompt == "first" else 0.01)
        yield f"done {prompt}"

    app = HarnessApp(HarnessConfig(), replier=replier, store=SessionStore(db))
    async with app.run_test() as pilot:
        for p in ["first", "second", "third"]:
            await submit(app, pilot, p)
        await pilot.pause(0.1)
        await pilot.press("ctrl+c")  # pauses with the rest still queued; then the app goes away
        await pilot.pause(0.1)
        sid = app.session_id
    assert ran == ["first"]

    ran.clear()
    app2 = HarnessApp(HarnessConfig(), replier=replier, store=SessionStore(db), resume=sid)
    async with app2.run_test() as pilot:
        await pilot.pause(0.3)
        assert app2.queue.pending == ("second", "third") and app2.queue.paused
        assert ran == [] and not app2.generating  # never auto-runs
        assert "2 queued prompts restored; Enter resumes, Esc clears" in shown(app2)
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert ran == ["second", "third"]


async def test_resume_restores_history_and_estimator_and_continue_picks_the_latest(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    app = HarnessApp(HarnessConfig(), replier=fast_reply, store=SessionStore(db))
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await app.workers.wait_for_complete()
        app.ledger.estimator.restore({"chars_per_token": 4.2, "chars": 420, "tokens": 100})
        app._save_session()
        sid = app.session_id

    for kwargs in ({"resume": sid[:4]}, {"continue_last": True}):
        app2 = HarnessApp(HarnessConfig(), replier=fast_reply, store=SessionStore(db), **kwargs)
        async with app2.run_test() as pilot:
            await pilot.pause(0.2)
            assert app2.session_id == sid
            assert [m["role"] for m in app2.fake_messages] == ["user", "assistant"]
            assert app2.ledger.estimator.chars_per_token == pytest.approx(4.2)
            assert "you: hello" in shown(app2) and [r.text for r in app2.query(Reply)] == ["echo hello"]


async def test_resume_of_an_unknown_id_starts_a_new_session_with_a_note() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply, resume="nope")
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert "no single session matches 'nope'" in shown(app)


async def test_fork_then_sessions_lists_both_and_they_diverge() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await app.workers.wait_for_complete()
        original = app.session_id
        await submit(app, pilot, "/fork")
        await pilot.pause(0.3)
        fork = app.session_id
        assert fork != original and app.store.get(fork).parent_id == original  # type: ignore[union-attr]
        assert [m["role"] for m in app.fake_messages] == ["user", "assistant"]
        await submit(app, pilot, "only in the fork")
        await app.workers.wait_for_complete()
        assert len(app.store.messages(original)) == 2 and len(app.store.messages(fork)) == 4
        await submit(app, pilot, "/sessions")
        await pilot.pause(0.1)
        text = shown(app)
        assert original in text and fork in text and f"fork of {original}" in text
        await submit(app, pilot, f"/resume {original}")
        await pilot.pause(0.3)
        assert app.session_id == original and len(app.fake_messages) == 2
        assert [r.text for r in app.query(Reply)] == ["echo hello"]


async def test_new_starts_an_empty_session_and_resume_brings_the_old_one_back() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await app.workers.wait_for_complete()
        first = app.session_id
        await submit(app, pilot, "/new")
        await pilot.pause(0.2)
        assert app.session_id != first and app.fake_messages == [] and not list(app.query(Reply))
        await submit(app, pilot, "/resume " + first)
        await pilot.pause(0.2)
        assert app.session_id == first and len(app.fake_messages) == 2


async def test_inspector_opens_on_ctrl_t_with_an_empty_state_and_closes_on_escape() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        await pilot.pause(0.1)
        assert isinstance(app.screen, Inspector)
        assert "no calls recorded" in shown_screen(app)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, Inspector)
        await submit(app, pilot, "/inspect x")
        await pilot.pause(0.1)
        assert "usage: /inspect [n]" in shown(app)


def shown_screen(app: HarnessApp) -> str:
    return "\n".join(str(w.render()) for w in app.screen.query(Static))


async def test_inspector_shows_each_calls_payload_and_counts() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        for i, (est, got) in enumerate([(100, 110), (200, 190)]):
            body = {"model": "m", "messages": [{"role": "user", "content": f"payload-{i}"}], "options": {"num_ctx": 4096}}
            n = app.store.record_call(app.session_id, "m", 4096, body, est)
            app.store.complete_call(app.session_id, n, got, 5)
        await submit(app, pilot, "/inspect 1")
        await pilot.pause(0.2)
        assert isinstance(app.screen, Inspector)
        text = shown_screen(app)
        assert "payload-0" in text and "estimate 100" in text and "prompt_eval_count 110" in text and "9.1% off" in text
        options = app.screen.query_one(OptionList)
        assert options.option_count == 2
        await pilot.press("down")
        await pilot.pause(0.1)
        assert "payload-1" in shown_screen(app) and "payload-0" not in shown_screen(app)
