"""The Jev Decider (ADR 0014): OpenRouter's Decisions API, with fakes and no network."""

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from harness.approval import Advice
from harness.decide.jev import DecideSpendCeiling, JevDecider
from harness.decide.log import Decision, DecisionLog
from harness.decide.questions import POLICY, check_policy
from harness.decide.tev import Choice, Noul, TevInputTooLarge

URL = "https://openrouter.ai/api/alpha/decisions"
KEY = "sk-or-v1-fake-key-for-tests"  # gitleaks:allow
REPLY = {
    "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
    "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
    "answers": {
        "safe": {"type": "noul", "noul": 0.93},
        "next": {"type": "choice", "choice": "retry", "confidence": 0.7, "probabilities": {"retry": 0.8, "ask": 0.2}},
    },
}


@respx.mock
async def test_ask_sends_the_documented_shape_and_logs_cost_and_the_served_model() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=REPLY))
    log = DecisionLog()
    jev = JevDecider(KEY, log=log)
    out = await jev.ask("t", {"a": 1}, {"safe": Noul("ok?"), "next": Choice("what?", {"retry": "again", "ask": None})})
    assert out["safe"].noul == 0.93 and out["next"].choice == "retry"
    sent = r.calls.last.request
    body = json.loads(sent.content)
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "~typesafe/jev-latest" and body["state"] == '{"a": 1}'
    assert body["questions"]["next"] == {"type": "choice", "instructions": "what?", "criteria": {"retry": "again", "ask": None}}
    [d] = log.entries()
    assert d.model == "typesafe/jev-1.13-20260917" and d.cost_usd == 0.00002 and d.input_tokens == 476
    assert jev.cost_usd == 0.00002 and "$0.000020" in d.line()


@respx.mock
async def test_a_failure_is_logged_without_the_key_and_raised() -> None:
    respx.post(URL).mock(return_value=httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}}))
    log = DecisionLog()
    with pytest.raises(httpx.HTTPStatusError):
        await JevDecider(KEY, log=log).ask("t", "x", POLICY)
    [d] = log.entries()
    assert "HTTPStatusError" in d.error and KEY not in d.error and "failed" in d.line()


@respx.mock
async def test_a_slow_answer_times_out_so_advice_never_stalls_a_turn() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json=REPLY)

    respx.post(URL).mock(side_effect=slow)
    log = DecisionLog()
    with pytest.raises(asyncio.TimeoutError):
        await JevDecider(KEY, log=log, timeout_s=0.05).ask("t", "x", POLICY)
    assert "TimeoutError" in log.entries()[0].error


@respx.mock
async def test_past_the_session_ceiling_nothing_more_is_sent() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=REPLY))
    log = DecisionLog()
    jev = JevDecider(KEY, log=log, max_cost_usd=0.00002)
    await jev.ask("t", "x", POLICY)
    with pytest.raises(DecideSpendCeiling):
        await jev.ask("t", "x", POLICY)
    assert r.call_count == 1 and "DecideSpendCeiling" in log.entries()[1].error


@respx.mock
async def test_an_oversized_state_is_refused_not_sent() -> None:
    r = respx.post(URL).mock(return_value=httpx.Response(200, json=REPLY))
    with pytest.raises(TevInputTooLarge):
        await JevDecider(KEY, max_input_tokens=100).ask("t", "word " * 500, POLICY)
    assert r.call_count == 0


@respx.mock
async def test_the_policy_check_works_through_jev() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=REPLY))
    assert await check_policy(JevDecider(KEY), "shell", {"command": "ls"}) == 0.93


def test_the_advice_line_names_its_source() -> None:
    assert Advice(p_safe=0.9, source="jev").line() == "jev: safe p=0.90 (advisory)"
    assert Advice(error="TimeoutError", source="jev").line() == "jev: unavailable (TimeoutError)"
    assert Advice(p_safe=0.2).line().startswith("tev1: risky")


def test_a_log_from_before_the_api_deciders_gets_the_cost_column(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, kind TEXT NOT NULL, "
               "model TEXT NOT NULL, state TEXT NOT NULL, summarized INTEGER NOT NULL, input_tokens INTEGER, "
               "answers TEXT NOT NULL, error TEXT NOT NULL DEFAULT '')")
    db.execute("INSERT INTO decisions (at, kind, model, state, summarized, answers) VALUES ('2026-10-01T00:00:00+00:00', 'policy', 'tev1', 's', 0, '{}')")
    db.commit()
    db.close()
    log = DecisionLog(path)
    log.record(Decision("policy", "jev", "s", {}, cost_usd=0.5))
    assert [d.cost_usd for d in log.entries()] == [None, 0.5]
