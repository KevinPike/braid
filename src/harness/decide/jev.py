"""Client for OpenRouter's Decisions API with Jev (ADR 0014).

Same shape as tev1's ``/v1/systemone`` (Choice, Noul and Score questions about a state, answers with probabilities), so
it reuses the question and answer types and the Decision log. Jev has a 32K window, bills input tokens only and reports
``usage.cost``; the session's total is kept here and, past ``max_cost_usd``, later calls raise instead of spending.
This module never imports ``anthropic``: it is plain HTTP.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx

from harness.decide.log import Decision, DecisionLog
from harness.decide.tev import MAX_QUESTIONS, Answer, Question, TevInputTooLarge, estimate_tokens, render_state

DEFAULT_MODEL = "~typesafe/jev-latest"
BASE_URL = "https://openrouter.ai/api"
MAX_INPUT_TOKENS = 8000  # Jev's window is 32K; a decision about a state needs far less, and input is what costs


class DecideSpendCeiling(RuntimeError):
    """The Decider has spent its ``max_cost_usd`` for this session; it stays off until the session ends."""


class JevDecider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = BASE_URL,
        log: DecisionLog | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        timeout_s: float = 4.0,
        max_cost_usd: float = 0.05,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.log = log
        self.max_input_tokens = max_input_tokens
        self.timeout_s = timeout_s
        self.max_cost_usd = max_cost_usd
        self.cost_usd = 0.0
        self._key = api_key
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 3.0)))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ask(
        self, kind: str, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Question]
    ) -> dict[str, Answer]:
        text = render_state(state)
        cost: float | None = None
        try:
            if self.cost_usd >= self.max_cost_usd:
                raise DecideSpendCeiling(f"decider spent ${self.cost_usd:.4f} of its ${self.max_cost_usd:.2f} session ceiling")
            if not 1 <= len(questions) <= MAX_QUESTIONS:
                raise ValueError(f"Jev takes 1 to {MAX_QUESTIONS} questions, got {len(questions)}")
            body_questions = {name: q.to_json() for name, q in questions.items()}
            size = estimate_tokens(text) + estimate_tokens(json.dumps(body_questions))
            if size > self.max_input_tokens:
                raise TevInputTooLarge(f"state and questions are ~{size} tokens, over the {self.max_input_tokens} cap")
            resp = await asyncio.wait_for(self._http.post(
                f"{self.base_url}/alpha/decisions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self.model, "state": text, "questions": body_questions},
            ), self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.cost_usd += float(cost)
            answers = {name: Answer.from_json(a) for name, a in data["answers"].items()}
        except Exception as exc:
            self._record(Decision(kind, self.model, text, {}, error=f"{type(exc).__name__}: {_scrub(exc, self._key)}", cost_usd=cost))
            raise
        self._record(Decision(
            kind, data.get("model") or self.model, text, {n: a.to_json() for n, a in answers.items()},
            input_tokens=usage.get("input_tokens"), cost_usd=cost,
        ))
        return answers

    def _record(self, decision: Decision) -> None:
        if self.log is not None:
            self.log.record(decision)


def _scrub(exc: Exception, key: str) -> str:
    return str(exc).replace(key, "***") if key else str(exc)
