"""Client for Ollama's ``/v1/systemone`` decision endpoint (tev1).

tev1 has a ~2K window fixed by the model, so the input is capped and an oversized state is
summarized first, never sent raw. Answers are probabilities; callers treat them as advice.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from harness.decide.log import Decision, DecisionLog

MAX_INPUT_TOKENS = 1500
# Deliberately pessimistic: the cap must hold for code and non-English text, which tokenize worse than prose.
CHARS_PER_TOKEN = 3.0
MAX_QUESTIONS = 64

Summarize = Callable[[str, int], Awaitable[str]]  # (text, budget in tokens) -> shorter text


class TevInputTooLarge(RuntimeError):
    """The state would not fit the cap even after summarizing."""


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str | None]

    def to_json(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Noul:
    """True/false; the answer is the probability of true."""

    instructions: str
    criteria: Mapping[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            out["criteria"] = dict(self.criteria)
        return out


@dataclass(frozen=True)
class Score:
    instructions: str
    levels: tuple[str, ...]  # lowest first

    def to_json(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


Question = Choice | Noul | Score


@dataclass(frozen=True)
class Answer:
    type: str
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None

    @staticmethod
    def from_json(obj: Mapping[str, Any]) -> Answer:
        return Answer(
            obj["type"], obj.get("choice"), obj.get("noul"), obj.get("score"),
            obj.get("probabilities") or {}, obj.get("confidence"),
        )

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {"type": self.type}
        for key in ("choice", "noul", "score", "confidence"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.probabilities:
            out["probabilities"] = dict(self.probabilities)
        return out


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def render_state(state: str | Mapping[str, Any] | list[Any]) -> str:
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True)


class TevClient:
    def __init__(
        self,
        host: str,
        model: str = "tev1:0.8b",
        *,
        log: DecisionLog | None = None,
        summarize: Summarize | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        keep_alive: str = "5m",
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.log = log
        self._summarize = summarize
        self.max_input_tokens = max_input_tokens
        self.keep_alive = keep_alive
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ask(
        self, kind: str, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Question]
    ) -> dict[str, Answer]:
        """Ask up to 64 questions about ``state``; every call, failed ones too, lands in the Decision log."""
        text = render_state(state)
        summarized = False
        try:
            if not 1 <= len(questions) <= MAX_QUESTIONS:
                raise ValueError(f"tev1 takes 1 to {MAX_QUESTIONS} questions, got {len(questions)}")
            body_questions = {name: q.to_json() for name, q in questions.items()}
            fixed = estimate_tokens(json.dumps(body_questions))
            budget = self.max_input_tokens - fixed
            if budget <= 0:
                raise TevInputTooLarge(f"the questions alone are ~{fixed} tokens, over the {self.max_input_tokens} cap")
            if estimate_tokens(text) > budget:
                if self._summarize is None:
                    raise TevInputTooLarge(f"state is ~{estimate_tokens(text)} tokens, over the {budget} left under the cap")
                text = await self._summarize(text, budget)
                summarized = True
                if estimate_tokens(text) > budget:
                    raise TevInputTooLarge(f"state is still ~{estimate_tokens(text)} tokens after summarizing (budget {budget})")
            resp = await self._http.post(f"{self.host}/v1/systemone", json={
                "model": self.model, "state": text, "questions": body_questions, "keep_alive": self.keep_alive,
            })
            resp.raise_for_status()
            data = resp.json()
            answers = {name: Answer.from_json(a) for name, a in data["answers"].items()}
        except Exception as exc:
            self._record(Decision(kind, self.model, text, {}, summarized, error=f"{type(exc).__name__}: {exc}"))
            raise
        self._record(Decision(
            kind, self.model, text, {n: a.to_json() for n, a in answers.items()}, summarized,
            (data.get("usage") or {}).get("input_tokens"),
        ))
        return answers

    def _record(self, decision: Decision) -> None:
        if self.log is not None:
            self.log.record(decision)
