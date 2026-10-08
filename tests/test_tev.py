import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from harness.app import HarnessApp
from harness.config import HarnessConfig
from harness.context.compaction import Compactor
from harness.context.ledger import CallRecord, Ledger
from harness.context.trimlog import TrimLog
from harness.decide.log import DecisionLog
from harness.decide.questions import ROUTER, compaction_state, route, should_compact
from harness.decide.tev import Choice, Noul, Score, TevClient, TevInputTooLarge, estimate_tokens
from textual.widgets import Input, Static

HOST = "http://ollama.test"
URL = f"{HOST}/v1/systemone"

ROUTE_REPLY = {"model": "tev1:0.8b", "usage": {"input_tokens": 120, "output_tokens": 3}, "answers": {
    "route": {"type": "choice", "choice": "tool", "probabilities": {"chat": 0.2, "tool": 0.8}, "confidence": 0.4}}}


@respx.mock
async def test_ask_sends_the_documented_shape_and_logs_probabilities() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=ROUTE_REPLY))
    log = DecisionLog()
    tev = TevClient(HOST, log=log)
    choice, p = await route(tev, "list my files")
    assert (choice, p) == ("tool", 0.8)
    body = json.loads(r.calls.last.request.content)
    assert body["model"] == "tev1:0.8b" and body["state"] == "User request: list my files"
    assert body["questions"]["route"]["type"] == "choice" and set(body["questions"]["route"]["criteria"]) == {"chat", "tool"}
    [d] = log.entries()
    assert d.kind == "router" and d.input_tokens == 120 and d.answers["route"]["probabilities"] == {"chat": 0.2, "tool": 0.8}
    assert "route=tool (0.80)" in d.line()


@respx.mock
async def test_noul_and_score_questions_round_trip() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"answers": {
        "a": {"type": "noul", "noul": 0.9}, "b": {"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}}}}))
    out = await TevClient(HOST).ask("t", "x", {"a": Noul("?"), "b": Score("?", ("low", "mid", "high"))})
    assert out["a"].noul == 0.9 and out["b"].score == 1.4


@respx.mock
async def test_input_over_the_cap_is_summarized_first_never_sent_raw() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=ROUTE_REPLY))
    seen: list[tuple[int, int]] = []

    async def shorten(text: str, budget: int) -> str:
        seen.append((len(text), budget))
        return "short version of the request"

    log = DecisionLog()
    tev = TevClient(HOST, log=log, summarize=shorten)
    huge = "word " * 5000
    await route(tev, huge)
    sent = json.loads(r.calls.last.request.content)["state"]
    assert sent == "short version of the request" and seen and "word word" not in sent
    questions_tokens = estimate_tokens(json.dumps({n: q.to_json() for n, q in ROUTER.items()}))
    assert estimate_tokens(sent) + questions_tokens <= 1500
    assert log.entries()[0].summarized


@respx.mock
async def test_without_a_summarizer_oversized_input_is_refused_not_sent() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=ROUTE_REPLY))
    log = DecisionLog()
    with pytest.raises(TevInputTooLarge):
        await route(TevClient(HOST, log=log), "word " * 5000)
    assert r.call_count == 0 and "TevInputTooLarge" in log.entries()[0].error


@respx.mock
async def test_a_summary_that_is_still_too_long_is_refused() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=ROUTE_REPLY))

    async def useless(text: str, budget: int) -> str:
        return text

    with pytest.raises(TevInputTooLarge, match="still"):
        await route(TevClient(HOST, summarize=useless), "word " * 5000)
    assert r.call_count == 0


@respx.mock
async def test_http_errors_are_logged_and_raised() -> None:
    respx.post(URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    log = DecisionLog()
    with pytest.raises(httpx.HTTPStatusError):
        await route(TevClient(HOST, log=log), "hi")
    assert "failed" in log.entries()[0].line()


async def test_question_count_is_validated() -> None:
    with pytest.raises(ValueError, match="1 to 64"):
        await TevClient(HOST).ask("t", "x", {})


def test_decision_log_persists(tmp_path):  # type: ignore[no-untyped-def]
    from harness.decide.log import Decision

    log = DecisionLog(tmp_path / "d.db")
    log.record(Decision("router", "m", "state", {"route": {"type": "choice", "choice": "chat", "probabilities": {"chat": 0.6}}}))
    log.close()
    [d] = DecisionLog(tmp_path / "d.db").entries()
    assert d.kind == "router" and d.id == 1 and d.at is not None


# --- compaction advice ---------------------------------------------------------------------------


def advised_session(p: float, used: int = 3000, *, fail: bool = False) -> tuple[Compactor, list[tuple[int, int, int, int]]]:
    asked: list[tuple[int, int, int, int]] = []
    ledger = Ledger(lambda _m: 4000)
    ledger.records.append(CallRecord(4000, {}, used, None, used, 0))

    async def advisor(u: int, n: int, g: int, t: int) -> float:
        asked.append((u, n, g, t))
        if fail:
            raise RuntimeError("tev down")
        return p

    async def summarize(_m):  # type: ignore[no-untyped-def]
        return "- gist"

    return Compactor(trimlog=TrimLog(), ledger=ledger, summarize=summarize, num_ctx=lambda: 4000, advisor=advisor), asked


async def test_tev_can_bring_compaction_forward_between_80_percent_and_the_ceiling() -> None:
    c, asked = advised_session(0.9, used=3000)  # 75%: below the point where tev1 is asked
    assert await c.advice() is None and asked == []
    c, asked = advised_session(0.9, used=3200)  # 80%
    c.policy = type(c.policy)(ceiling=0.99)  # so the projection alone would not compact yet
    assert c.should_compact() is None
    assert await c.advice() == "decision layer advised compaction (p=0.90)"
    assert asked[0][:2] == (3200, 4000)


async def test_tev_saying_no_or_failing_never_blocks_the_hard_rule() -> None:
    c, _ = advised_session(0.1, used=3200)
    c.policy = type(c.policy)(ceiling=0.99)
    assert await c.advice() is None
    c, _ = advised_session(0.9, used=3200, fail=True)
    c.policy = type(c.policy)(ceiling=0.99)
    assert await c.advice() is None  # failure is swallowed
    c, _ = advised_session(0.1, used=3500)
    assert c.should_compact() is not None  # ceiling decides alone


@respx.mock
async def test_compaction_question_state_is_plain_numbers() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json={"answers": {"compact": {"type": "noul", "noul": 0.7}}}))
    assert await should_compact(TevClient(HOST), used=3300, num_ctx=4000, growth=500, turns=12) == 0.7
    state = json.loads(r.calls.last.request.content)["state"]
    assert state == compaction_state(used=3300, num_ctx=4000, growth=500, turns=12) and "82%" in state


# --- TUI -------------------------------------------------------------------------------------


async def _one(prompt: str) -> AsyncIterator[str]:
    yield prompt


async def test_slash_decisions_lists_the_log() -> None:
    from harness.decide.log import Decision

    log = DecisionLog()
    log.record(Decision("router", "tev1:0.8b", "User request: hi", {"route": {"type": "choice", "choice": "chat", "probabilities": {"chat": 0.7}}}))
    app = HarnessApp(HarnessConfig(), replier=_one, decisions=log)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "/decisions"
        await pilot.press("enter")
        await pilot.pause(0.2)
        shown = " ".join(str(w.render()) for w in app.query(Static))
    assert "router: route=chat (0.70)" in shown


@respx.mock
async def test_each_prompt_is_routed_and_logged_without_blocking_the_reply() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=ROUTE_REPLY))
    log = DecisionLog()
    app = HarnessApp(HarnessConfig(), replier=_one, decisions=log, tev=TevClient(HOST, log=log))
    async with app.run_test() as pilot:
        app.query_one(Input).value = "list my files"
        await pilot.press("enter")
        await pilot.pause(0.3)
    assert [d.kind for d in log.entries()] == ["router"]
