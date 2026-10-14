"""Structured model calls: one schema-shaped JSON answer per call.

Ollama goes through the guard client (ADR 0004); Claude goes through the Anthropic API (ADR 0006).
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

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


def openrouter_route(response: Reply) -> tuple[str, str]:
    """(provider, task type) from an OpenRouter reply: ``provider`` is a top-level field, and ``openrouter/auto``'s classifier
    puts ``task_type`` in its pipeline step when the request asked for ``X-OpenRouter-Metadata``. Empty when absent."""
    extra = response.model_extra or {}
    meta = extra.get("openrouter_metadata")
    steps = meta.get("pipeline") if isinstance(meta, dict) else None
    task = next((str(d["task_type"]) for s in steps or [] if isinstance(s, dict) and isinstance(d := s.get("data"), dict) and "task_type" in d), "")
    return str(extra.get("provider") or ""), task


def warn_slow(model_id: str, seconds: float) -> None:
    """The default ``on_slow``: one permanent line on stderr (over a live spinner line if there is one)."""
    clear = "\r\x1b[K" if sys.stderr.isatty() else ""
    sys.stderr.write(f"{clear}  ⚠ slow call: {model_id} has taken {seconds:.0f}s and is still going\n")
    sys.stderr.flush()


# F10: server-side refusal fallbacks, for the models that offer them on the Claude API. Batches reject them.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5"})

Reply = anthropic.types.Message | anthropic.types.beta.BetaMessage


class AnthropicWorkflowModel:
    """Asks the Anthropic API for JSON in ``schema``'s shape (ADR 0006). Never sends sampling parameters.

    Each request's thinking summary (F9, ``thinking_display``) is kept on its ``ApiCall`` in ``calls``, so a node's run
    log line carries it the way it carries the node's usage. ``fallbacks`` (F10, the Arbiter's Sonnet or Opus 5.5 only)
    sends the request on the beta endpoint with ``fallbacks: "default"``, so a refusal is retried on the server by a
    substitute model; each ``ApiCall`` records the model that served it.

    OpenRouter (ADR 0009) is the same class over a client pointed at its Anthropic-shaped endpoint: ``extra_body`` adds its
    routing fields, ``reported_cost`` takes each call's price from the response's ``usage.cost`` (so no ``PRICING`` entry
    is needed), and ``vet_served`` may refuse the model that answered (``openrouter/auto`` picks its own)."""

    def __init__(
        self, client: anthropic.AsyncAnthropic, model_id: str, *, max_tokens: int = 16000, effort: Effort | None = None,
        thinking_display: ThinkingDisplay = "summarized", fallbacks: bool = False, extra_body: dict[str, Any] | None = None,
        reported_cost: bool = False, vet_served: Callable[[str], None] | None = None, thinking_sample: float = 1.0,
        rng: random.Random | None = None, slow_after_s: float | None = None,
        on_slow: Callable[[str, float], None] | None = None,
    ) -> None:
        if not reported_cost:
            require_pricing(model_id)
        self.thinking_sample = thinking_sample  # the share of complete() calls that think; the rest send thinking disabled
        self.rng = rng or random.Random()
        self.slow_after_s = slow_after_s  # warn once when one request takes this long (None: never)
        self.on_slow = on_slow or warn_slow
        self.extra_body = extra_body
        self.reported_cost = reported_cost
        self.vet_served = vet_served
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking_param(thinking_display)
        self.fallbacks = fallbacks
        self.calls: list[ApiCall] = []

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def _record(self, response: Reply, thinking_on: bool | None = None, seconds: float | None = None) -> None:
        u = response.usage
        cost: float | None = None
        provider, task_type = openrouter_route(response) if self.reported_cost else ("", "")
        if self.reported_cost:  # OpenRouter adds ``cost`` (USD) to ``usage``; the SDK keeps it as an extra field
            reported = getattr(u, "cost", None)
            if not isinstance(reported, (int, float)):  # the call is billed, so keep its tokens (at 0: unknown) and stop the run
                self.calls.append(api_call(
                    self.model_id, u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens,
                    response._request_id, served_model=str(response.model), cost_usd=0.0, thinking_on=thinking_on,
                    stop_reason=str(response.stop_reason or ""), seconds=seconds, provider=provider, task_type=task_type,
                ))
                raise ApiFailure("UnreportedCost", f"{self.model_id}: the reply's usage carried no cost", request_id=response._request_id)
            cost = float(reported)
        self.calls.append(api_call(
            self.model_id, u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens,
            response._request_id, thinking=thinking_text(response.content), served_model=str(response.model), cost_usd=cost,
            thinking_on=thinking_on, stop_reason=str(response.stop_reason or ""), seconds=seconds,
            provider=provider, task_type=task_type,
        ))
        if self.vet_served is not None:
            self.vet_served(str(response.model))

    async def _create(self, params: dict[str, Any]) -> Reply:
        if self.fallbacks:  # ``fallbacks`` is a typed param of the beta endpoint in the SDK, so no ``extra_body``
            return cast(Reply, await self.client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default"))
        if self.extra_body:
            return cast(Reply, await self.client.messages.create(**params, extra_body=self.extra_body))
        return cast(Reply, await self.client.messages.create(**params))

    async def _timed(self, params: dict[str, Any]) -> tuple[Reply, float]:
        """``_create`` and its wall time; warns once (``on_slow``) when it passes ``slow_after_s`` and goes on waiting."""
        start = time.monotonic()
        task = asyncio.ensure_future(self._create(params))
        try:
            if self.slow_after_s:
                done, _ = await asyncio.wait({task}, timeout=self.slow_after_s)
                if not done:
                    self.on_slow(self.model_id, time.monotonic() - start)
            reply = await task
        finally:
            if not task.done():  # cancelled (Ctrl-C) while waiting
                task.cancel()
        return reply, time.monotonic() - start

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        # `messages.create` rather than `messages.parse`: parse validates inside the HTTP layer, so an invalid or
        # truncated reply raises before `stop_reason` and `usage` can be read.
        output_config: anthropic.types.OutputConfigParam = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        if self.effort is not None:
            output_config["effort"] = self.effort
        thinking_on: bool | None = None
        extra: dict[str, Any] = {"thinking": self.thinking} if self.thinking else {}
        if self.thinking_sample < 1.0:  # one draw per call, shared by its retry
            thinking_on = self.rng.random() < self.thinking_sample
            if not thinking_on:
                extra = {"thinking": {"type": "disabled"}}
        error = ""
        for _ in range(2):  # one retry, with the problem appended
            try:
                response, seconds = await self._timed({
                    "model": self.model_id,
                    "max_tokens": self.max_tokens,
                    "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": prompt + error}],
                    "output_config": output_config,
                    **extra,
                })
            except API_FAULTS as exc:  # past the SDK's retries: the run stops here, resumable (F8)
                raise ApiFailure.from_exc(exc) from exc
            # a 200 whose body is not JSON comes back as text
            if not isinstance(response, (anthropic.types.Message, anthropic.types.beta.BetaMessage)):
                raise ApiFailure("UnexpectedBody", f"the reply was not a message: {str(response)[:100]!r}")
            self._record(response, thinking_on, seconds)
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
