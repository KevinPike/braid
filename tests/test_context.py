import json
import random

import httpx
import respx
from strands import Agent

from harness.agent import GuardedOllamaModel, stream_reply
from harness.context.accounting import Category, TokenEstimator, account
from harness.context.ledger import Ledger
from harness.guard.ollama_client import CallMetrics, OllamaClient
from harness.guard.watchdog import Level, Watchdog

HOST = "http://ollama.test"


def msgs(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in pairs]


def test_account_splits_categories() -> None:
    m = msgs(("system", "s" * 10), ("user", "u1"), ("assistant", "a1"), ("user", "now"))
    m.insert(4, {"role": "tool", "content": "t" * 7})
    a = account(m, [{"type": "function", "function": {"name": "f"}}])
    assert a.chars[Category.SYSTEM] == 10
    assert a.chars[Category.HISTORY] == len("u1") + len("a1")
    assert a.chars[Category.TURN] == 3
    assert a.chars[Category.TOOL_RESULTS] == 7
    assert a.chars[Category.TOOLS] > 0
    assert sum(a.messages.values()) == 5


def fake_tokens(messages: list[dict[str, str]]) -> int:
    """A stand-in tokenizer with a different ratio and framing than the estimator's defaults."""
    return sum(len(m["content"]) * 10 // 29 + 5 for m in messages)


def test_estimate_within_5pct_over_30_turns_and_categories_add_up() -> None:
    rng = random.Random(1)
    ledger = Ledger(lambda _m: 32768)
    history: list[dict[str, str]] = [{"role": "system", "content": "You are helpful. " * 20}]
    for turn in range(30):
        history.append({"role": "user", "content": "word " * rng.randint(5, 400)})
        ledger.before_call("m", history, None)
        ledger.after_call(CallMetrics(fake_tokens(history), 10, 1.0, 0.0))
        history.append({"role": "assistant", "content": "reply " * rng.randint(5, 200)})
    # the first calls teach the ratio; the rest must be inside the 5% gate
    for rec in ledger.records[3:]:
        assert rec.error is not None and rec.error < 0.05
        assert abs(sum(rec.scaled_categories().values()) - rec.prompt_eval_count) <= 0.05 * rec.prompt_eval_count  # type: ignore[operator]
        assert abs(sum(rec.categories.values()) - rec.estimate) == 0


def test_estimator_does_not_learn_when_nothing_to_learn() -> None:
    est = TokenEstimator()
    est.observe(account([], None), 100)
    assert est.chars_per_token == 3.5


@respx.mock
async def test_oversized_paste_fires_red_truncation_and_logs_gap(caplog: object) -> None:
    import logging

    num_ctx = 2048
    # Ollama silently keeps only num_ctx tokens of the prompt
    respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(200, content=json.dumps({
        "done": True, "message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": num_ctx,
        "eval_count": 3, "eval_duration": 10**9, "load_duration": 0, "total_duration": 10**9}).encode()))
    client = OllamaClient(HOST)
    client.pin("m", num_ctx)

    class NoPs:
        async def ps(self) -> list[object]:
            return []

    wd = Watchdog(NoPs(), pinned=client.pinned, memory_level=lambda: 1)  # type: ignore[arg-type]
    client.on_call(wd.record_call)
    ledger = Ledger(client.pinned)
    agent = Agent(model=GuardedOllamaModel(client, "m", ledger), system_prompt="sys", callback_handler=None)
    with caplog.at_level(logging.WARNING):  # type: ignore[attr-defined]
        _ = [d async for d in stream_reply(agent, "lorem ipsum " * 2000)]
    assert wd.state.level == Level.RED
    alert = next(a for a in wd.state.alerts if a.kind == "truncation")
    rec = ledger.records[-1]
    assert rec.gap is not None and rec.gap > 4000
    assert f"gap {rec.gap}" in alert.message
    assert f"gap {rec.gap} tokens" in caplog.text  # type: ignore[attr-defined]
    assert rec.strands_estimate is not None  # compared with Strands' own estimate


def test_context_panel_shows_the_tool_schema_cost() -> None:
    from harness.chat import render_context
    from harness.context.accounting import Category
    from harness.context.ledger import CallRecord

    cats = {c: 0 for c in Category}
    cats[Category.TOOLS] = 412
    rec = CallRecord(4096, cats, 412, None, 412, 5)
    assert "tools 412" in render_context(rec)
