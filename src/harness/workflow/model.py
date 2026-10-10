"""Structured model calls: one schema-shaped JSON answer per call.

Ollama goes through the guard client (ADR 0004); Claude goes through the Anthropic API (ADR 0006).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from harness.guard.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class WorkflowModelError(RuntimeError):
    """The model did not return a valid reply for the schema, even after one retry."""


class WorkflowModel(Protocol):
    async def complete(self, system: str, prompt: str, schema: type[T]) -> T: ...


class GuardedWorkflowModel:
    """Asks ``OllamaClient.chat`` for JSON in ``schema``'s shape, so ``num_ctx`` stays pinned for every request."""

    def __init__(
        self, client: OllamaClient, model_id: str, *, temperature: float = 0.2, think: bool | None = None,
        num_predict: int | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.think = think
        self.num_predict = num_predict

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        error = ""
        for _ in range(2):  # one retry, with the validation error appended
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt + error}]
            options: dict[str, float | int] = {"temperature": self.temperature}
            if self.num_predict is not None:
                options["num_predict"] = self.num_predict
            parts: list[str] = []
            thought = False
            async for chunk in self.client.chat(
                self.model_id, messages, format=schema.model_json_schema(), options=options, think=self.think,
            ):
                parts.append(chunk.text)
                thought = thought or bool(chunk.thinking)  # thinking text is dropped
            text = "".join(parts)
            if not text.strip() and thought:
                error = (
                    "\n\nYour last reply had no content: the model returned only thinking text. "
                    "Reply with valid JSON only, and keep any reasoning short."
                )
                continue
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                error = (
                    f"\n\nYour last reply was invalid ({exc.error_count()} errors: {str(exc)[:300]}). "
                    "Reply with valid JSON only."
                )
        raise WorkflowModelError(f"{self.model_id} returned invalid JSON twice for {schema.__name__}:{error}")


# USD per million tokens (input, output). Cache reads cost 0.1x input, cache writes 1.25x input.
PRICING: dict[str, tuple[float, float]] = {"claude-haiku-5-5": (0.10, 0.50)}


@dataclass(frozen=True)
class ApiCall:
    """Usage and cost of one Anthropic API request."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
    request_id: str | None


class AnthropicWorkflowModel:
    """Asks the Anthropic API for JSON in ``schema``'s shape (ADR 0006). Never sends sampling parameters."""

    def __init__(
        self, client: anthropic.AsyncAnthropic, model_id: str, *, max_tokens: int = 16000, effort: Effort | None = None,
    ) -> None:
        if model_id not in PRICING:
            raise ValueError(f"no pricing for {model_id!r}; add it to PRICING (known: {', '.join(PRICING)})")
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.calls: list[ApiCall] = []

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def _record(self, response: anthropic.types.Message) -> None:
        u = response.usage
        cache_read = u.cache_read_input_tokens or 0
        cache_write = u.cache_creation_input_tokens or 0
        price_in, price_out = PRICING[self.model_id]
        cost = (u.input_tokens * price_in + cache_read * price_in * 0.1 + cache_write * price_in * 1.25
                + u.output_tokens * price_out) / 1_000_000
        self.calls.append(ApiCall(u.input_tokens, u.output_tokens, cache_read, cache_write, cost, response._request_id))

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        # `messages.create` rather than `messages.parse`: parse validates inside the HTTP layer, so an invalid or
        # truncated reply raises before `stop_reason` and `usage` can be read.
        output_config: anthropic.types.OutputConfigParam = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        if self.effort is not None:
            output_config["effort"] = self.effort
        error = ""
        for _ in range(2):  # one retry, with the problem appended
            response = await self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt + error}],
                output_config=output_config,
            )
            self._record(response)
            if response.stop_reason == "refusal":
                category = response.stop_details.category if response.stop_details else None
                raise WorkflowModelError(f"{self.model_id} refused the request (category: {category})")
            if response.stop_reason == "max_tokens":
                error = "\n\nYour last reply was cut off at the token limit. Reply with valid JSON only, and more briefly."
                continue
            text = "".join(b.text for b in response.content if b.type == "text")
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                error = (
                    f"\n\nYour last reply was invalid ({exc.error_count()} errors: {str(exc)[:300]}). "
                    "Reply with valid JSON only."
                )
        raise WorkflowModelError(f"{self.model_id} returned invalid JSON twice for {schema.__name__}:{error}")
