"""Structured model calls: one schema-shaped JSON answer per call.

Ollama goes through the guard client (ADR 0004); Claude goes through the Anthropic API (ADR 0006).
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from harness.anthropic_api import API_FAULTS, ApiFailure, thinking_param, thinking_text
from harness.config import Effort, ThinkingDisplay
from harness.guard.ollama_client import OllamaClient
from harness.usage import PRICING, ApiCall, Price, api_call, require_pricing

T = TypeVar("T", bound=BaseModel)


class WorkflowModelError(RuntimeError):
    """The model did not return a valid reply for the schema, even after one retry. A failed request is not this: it
    raises ``ApiFailure``, which no node catches, so the run stops with status ``api_error`` (F8)."""


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


class AnthropicWorkflowModel:
    """Asks the Anthropic API for JSON in ``schema``'s shape (ADR 0006). Never sends sampling parameters.

    Each request's thinking summary (F9, ``thinking_display``) is kept on its ``ApiCall`` in ``calls``, so a node's run
    log line carries it the way it carries the node's usage."""

    def __init__(
        self, client: anthropic.AsyncAnthropic, model_id: str, *, max_tokens: int = 16000, effort: Effort | None = None,
        thinking_display: ThinkingDisplay = "summarized",
    ) -> None:
        require_pricing(model_id)
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking_param(thinking_display)
        self.calls: list[ApiCall] = []

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def _record(self, response: anthropic.types.Message) -> None:
        u = response.usage
        self.calls.append(api_call(
            self.model_id, u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens,
            response._request_id, thinking=thinking_text(response.content),
        ))

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        # `messages.create` rather than `messages.parse`: parse validates inside the HTTP layer, so an invalid or
        # truncated reply raises before `stop_reason` and `usage` can be read.
        output_config: anthropic.types.OutputConfigParam = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        if self.effort is not None:
            output_config["effort"] = self.effort
        extra: dict[str, Any] = {"thinking": self.thinking} if self.thinking else {}
        error = ""
        for _ in range(2):  # one retry, with the problem appended
            try:
                response = await self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": prompt + error}],
                    output_config=output_config,
                    **extra,
                )
            except API_FAULTS as exc:  # past the SDK's retries: the run stops here, resumable (F8)
                raise ApiFailure.from_exc(exc) from exc
            if not isinstance(response, anthropic.types.Message):  # a 200 whose body is not JSON comes back as text
                raise ApiFailure("UnexpectedBody", f"the reply was not a message: {str(response)[:100]!r}")
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


__all__ = [
    "PRICING", "AnthropicWorkflowModel", "ApiCall", "ApiFailure", "GuardedWorkflowModel", "Price", "WorkflowModel",
    "WorkflowModelError",
]
