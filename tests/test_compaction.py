import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from strands.types.content import Message
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.config import HarnessConfig
from harness.context.accounting import TokenEstimator
from harness.context.compaction import SUMMARY_HEADER, CompactionPolicy, Compactor, message_text
from harness.context.ledger import CallRecord, Ledger
from harness.context.strategies import LoggedFileStorage, chunk_messages
from harness.context.trimlog import TrimEntry, TrimLog

NUM_CTX = 4000


def text(role: str, body: str) -> Message:
    return {"role": role, "content": [{"text": body}]}  # type: ignore[typeddict-item]


class Session:
    """A scripted chat against a fake model: prompt tokens are measured the way Ollama would (chars / 4 plus framing)."""

    def __init__(self, *, policy: CompactionPolicy = CompactionPolicy(), summary: str = "- gist") -> None:
        self.estimator = TokenEstimator(4.0)
        self.ledger = Ledger(lambda _m: NUM_CTX, self.estimator)
        self.trimlog = TrimLog()
        self.messages: list[Message] = []
        self.summarized: list[list[Message]] = []
        self.summary = summary

        async def summarize(msgs: Sequence[Message]) -> str:
            self.summarized.append(list(msgs))
            facts = [line for m in msgs for line in message_text(m).splitlines() if "FACT" in line]
            return "\n".join([self.summary, *facts])

        self.compactor = Compactor(trimlog=self.trimlog, ledger=self.ledger, summarize=summarize,
                                   num_ctx=lambda: NUM_CTX, policy=policy)

    def prompt_tokens(self) -> int:
        return sum(len(message_text(m)) // 4 + 4 for m in self.messages) + 20  # + system prompt

    async def turn(self, user: str, assistant: str) -> int:
        self.messages.append(text("user", user))
        prompt = self.prompt_tokens()
        self.messages.append(text("assistant", assistant))
        reply = len(assistant) // 4
        self.ledger.records.append(CallRecord(NUM_CTX, {}, prompt, None, prompt, reply))
        await self.compactor.after_turn(self.messages)
        return prompt


async def test_hundred_turn_session_never_passes_the_ceiling() -> None:
    s = Session()
    worst = 0
    for i in range(100):
        worst = max(worst, await s.turn(f"question {i} " + "x" * 400, f"answer {i} " + "y" * 400))
    assert worst <= 0.9 * NUM_CTX
    assert len(s.trimlog.entries()) >= 5  # it really did have to compact, repeatedly


async def test_a_growing_turn_is_anticipated_from_the_biggest_turn_so_far() -> None:
    s = Session()
    for i in range(10):
        await s.turn("a" * 400, "b" * 400)
    worst = 0
    for i in range(40):  # turns now three times bigger than before
        worst = max(worst, await s.turn("a" * 1200, "b" * 1200))
    assert worst <= 0.9 * NUM_CTX


async def test_facts_from_turn_five_survive_compaction_in_order() -> None:
    s = Session()
    for i in range(60):
        fact = f"FACT-{i}: the code is {i * 111}" if i in (5, 6, 7, 8, 9) else ""
        await s.turn(f"{fact} turn {i} " + "x" * 400, "ok " + "y" * 400)
    kept = "\n".join(message_text(m) for m in s.messages)
    assert all(f"FACT-{i}" in kept for i in (5, 6, 7, 8, 9))
    assert SUMMARY_HEADER in kept
    assert message_text(s.messages[-1]).startswith("ok")  # newest turn untouched


async def test_pinned_messages_are_never_summarized() -> None:
    s = Session(policy=CompactionPolicy(pin_first=2))
    s.messages.extend([text("user", "PINNED-RULE: always answer in French"), text("assistant", "d'accord")])
    for i in range(40):
        await s.turn("x" * 400, "y" * 400)
    assert message_text(s.messages[0]).startswith("PINNED-RULE")
    assert not any("PINNED-RULE" in message_text(m) for batch in s.summarized for m in batch)


async def test_tool_call_and_result_are_never_split() -> None:
    s = Session(policy=CompactionPolicy(preserve_recent=2))
    use: Message = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "read", "input": {}}}]}
    result: Message = {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success",
                                                                   "content": [{"text": "z" * 3000}]}}]}
    s.messages.extend([text("user", "x" * 400), use, result, text("assistant", "y" * 400)])
    await s.compactor.compact(s.messages, "test")
    ids = [(b.get("toolUse") or b.get("toolResult"))["toolUseId"] for m in s.messages for b in m["content"]  # type: ignore[index]
           if "toolUse" in b or "toolResult" in b]
    assert ids in ([], ["t1", "t1"])


async def test_every_compaction_has_a_trim_log_entry() -> None:
    s = Session()
    for i in range(30):
        await s.turn("x" * 400, "y" * 400)
    entries = s.trimlog.entries()
    assert entries and all(e.strategy == "summarize" and e.messages_removed > 0 and e.tokens_after < e.tokens_before
                           for e in entries)
    assert "over 90% of num_ctx" in entries[0].reason


async def test_a_failing_summarizer_leaves_history_untouched_and_is_logged() -> None:
    s = Session()

    async def boom(_m: Sequence[Message]) -> str:
        raise RuntimeError("ollama down")

    s.compactor._summarize = boom
    for i in range(3):
        await s.turn("x" * 400, "y" * 400)
    before = list(s.messages)
    assert await s.compactor.compact(s.messages, "test") is None
    assert s.messages == before
    assert "summarizer failed" in s.trimlog.entries()[-1].detail


async def test_truncation_forces_compaction_even_under_the_ceiling() -> None:
    s = Session()
    for i in range(10):
        await s.turn("x" * 100, "y" * 100)
    assert s.compactor.should_compact() is None
    s.compactor.force("Truncation detected")
    assert await s.compactor.after_turn(s.messages) is not None
    assert s.trimlog.entries()[-1].reason == "Truncation detected"


def test_trim_log_persists_in_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    log = TrimLog(path)
    log.record(TrimEntry("summarize", "why", 4, 900, 300, "d"))
    log.close()
    [entry] = TrimLog(path).entries()
    assert (entry.strategy, entry.reason, entry.messages_removed, entry.tokens_freed) == ("summarize", "why", 4, 600)
    assert entry.at is not None and entry.id == 1


def test_chunks_fit_the_budget_and_keep_order() -> None:
    est = TokenEstimator(4.0)
    msgs = [text("user", f"{i:03d}" + "x" * 397) for i in range(20)]
    chunks = chunk_messages(msgs, 500, est)
    assert all(sum(est.tokens(len(message_text(m)) + 6, 1) for m in c) <= 500 or len(c) == 1 for c in chunks)
    assert [m for c in chunks for m in c] == msgs


async def test_offloading_a_tool_result_is_logged(tmp_path: Path) -> None:
    log = TrimLog()
    storage = LoggedFileStorage(str(tmp_path), log, TokenEstimator(4.0), preview_tokens=100)
    ref = await storage.store("k1", b"q" * 8000)
    [entry] = log.entries()
    assert entry.strategy == "offload" and entry.tokens_before == 2000 and ref in entry.detail
    assert (await storage.retrieve(ref))[0] == b"q" * 8000


# --- app: compaction runs between turns, and the trims are viewable -----------------------------


async def test_compaction_runs_between_turns_and_the_queue_waits_for_it() -> None:
    events: list[str] = []

    async def reply(prompt: str) -> AsyncIterator[str]:
        events.append(f"start {prompt}")
        await asyncio.sleep(0.05)
        events.append(f"end {prompt}")
        yield prompt

    s = Session()
    app = HarnessApp(HarnessConfig(), replier=reply, trimlog=s.trimlog, compactor=s.compactor)
    app.fake_messages = s.messages
    for i in range(8):
        s.messages.extend([text("user", "x" * 400), text("assistant", "y" * 400)])
    s.ledger.records.append(CallRecord(NUM_CTX, {}, 0, None, 3700, 100))

    original = s.compactor._summarize

    async def slow(msgs: Sequence[Message]) -> str:
        events.append("compact start")
        await asyncio.sleep(0.2)
        events.append("compact end")
        return await original(msgs)

    s.compactor._summarize = slow
    async with app.run_test() as pilot:
        for prompt in ("one", "two"):
            app.query_one(Input).value = prompt
            await pilot.press("enter")
        await pilot.pause(0.8)
    assert events[:4] == ["start one", "end one", "compact start", "compact end"]
    assert events.index("start two") > events.index("compact end")


async def test_slash_trims_lists_every_trim_in_the_tui() -> None:
    log = TrimLog()
    log.record(TrimEntry("summarize", "projected over ceiling", 6, 3500, 1200))
    log.record(TrimEntry("offload", "tool result too big", 0, 9000, 1000, "stored as k1"))
    app = HarnessApp(HarnessConfig(), replier=lambda p: _one(p), trimlog=log)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "/trims"
        await pilot.press("enter")
        await pilot.pause(0.2)
        shown = " ".join(str(w.render()) for w in app.query(Static))
    assert "summarize: projected over ceiling" in shown and "offload: tool result too big" in shown


async def test_new_trims_announce_themselves_in_the_chat() -> None:
    log = TrimLog()
    app = HarnessApp(HarnessConfig(), replier=lambda p: _one(p), trimlog=log)
    async with app.run_test() as pilot:
        log.record(TrimEntry("sliding_window", "history over 40 messages", 2, 800, 600))
        await pilot.pause(0.2)
        shown = " ".join(str(w.render()) for w in app.query(Static))
    assert "sliding_window: history over 40 messages" in shown


async def _one(prompt: str) -> AsyncIterator[str]:
    yield prompt


@pytest.mark.parametrize("strategy", ["none", "sliding_window", "summarize"])
def test_strategy_setting_is_validated(strategy: str) -> None:
    assert HarnessConfig.model_validate({"context": {"strategy": strategy}}).context.strategy == strategy


def test_sliding_window_trims_are_logged() -> None:
    from types import SimpleNamespace

    from harness.context.strategies import LoggedSlidingWindow

    log = TrimLog()
    manager = LoggedSlidingWindow(4, log, TokenEstimator(4.0))
    agent = SimpleNamespace(messages=[text("user" if i % 2 == 0 else "assistant", "m" * 80) for i in range(10)])
    manager.apply_management(agent)  # type: ignore[arg-type]
    assert len(agent.messages) <= 4
    [entry] = log.entries()
    assert entry.strategy == "sliding_window" and entry.messages_removed == 10 - len(agent.messages)
