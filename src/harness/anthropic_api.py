"""The Anthropic client, and what a request that fails past the SDK's own retries becomes (F8).

The SDK retries 408, 409, 429, 5xx (529 overloaded included) and connection errors with backoff, honoring
``retry-after``, ``[anthropic] max_retries`` times. Whatever still fails becomes one ``ApiFailure``: a Workflow run
ends with status ``api_error`` at that node (Checkpoint kept, nothing applied), and a chat turn ends with a note.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable
from typing import Any

import anthropic
import httpx2

from harness.config import AnthropicConfig, OpenRouterConfig, ThinkingDisplay

# What a request can raise once the SDK has given up: its own errors, a transport error mid-stream (the SDK wraps those
# only before the response starts), and a 200 whose body is not JSON (the SDK parses it with ``json.loads``).
API_FAULTS: tuple[type[BaseException], ...] = (anthropic.APIError, httpx2.TransportError, json.JSONDecodeError)

_KEY = re.compile(r"(sk-(?:ant|or)-)[A-Za-z0-9_\-]+")


class ApiFailure(RuntimeError):
    """An Anthropic API request that failed for good. ``kind`` is the SDK's error type (or what the harness found wrong
    with the reply); the message is cut short and never carries a key."""

    def __init__(self, kind: str, message: str, *, request_id: str | None = None, status_code: int | None = None) -> None:
        self.kind = kind
        self.message = _KEY.sub(r"\1…", message)[:300]
        self.request_id = request_id
        self.status_code = status_code
        super().__init__(f"{kind}: {self.message}")

    @classmethod
    def from_exc(cls, exc: BaseException) -> ApiFailure:
        status = getattr(exc, "status_code", None)
        return cls(
            type(exc).__name__, str(exc) or type(exc).__name__, request_id=getattr(exc, "request_id", None),
            status_code=status if isinstance(status, int) else None,
        )

    def record(self) -> dict[str, Any]:
        """The fields a run log line or a final state carries."""
        return {"error": self.kind, "status_code": self.status_code, "request_id": self.request_id, "message": self.message}

    def note(self) -> str:
        """One line for a human: the type, the HTTP status, the request id."""
        status = f" {self.status_code}" if self.status_code else ""
        request = f", request {self.request_id}" if self.request_id else ""
        return f"{self.kind}{status}{request}: {self.message}"


def find_failure(exc: BaseException) -> ApiFailure | None:
    """The ``ApiFailure`` behind ``exc``, which Strands wraps in its own ``EventLoopException``."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ApiFailure):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def make_client(config: AnthropicConfig) -> anthropic.AsyncAnthropic:
    """The client every API path uses, with the SDK's own credential resolution (a key is never passed in)."""
    return anthropic.AsyncAnthropic(
        max_retries=config.max_retries, timeout=httpx2.Timeout(config.timeout, connect=5.0),
    )


def make_openrouter_client(config: OpenRouterConfig) -> anthropic.AsyncAnthropic:
    """The OpenRouter client (ADR 0009): the Anthropic SDK pointed at OpenRouter's Anthropic-shaped endpoint. The key is
    ``OPENROUTER_API_KEY``, read from the environment here and nowhere else."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set (put it in .env and run with `uv run --env-file .env`)")
    return anthropic.AsyncAnthropic(
        base_url=config.base_url, api_key=key, max_retries=config.max_retries,
        timeout=httpx2.Timeout(config.timeout, connect=5.0), default_headers={"X-Title": "braid", "X-OpenRouter-Metadata": "enabled"},  # metadata: the auto-router's task type
    )


def openrouter_body(config: OpenRouterConfig, model_id: str, *, session_id: str | None = None) -> dict[str, Any]:
    """The OpenRouter fields a request adds to the Anthropic-shaped body (``extra_body``): provider routing, the
    fallback chain, and for ``openrouter/auto`` the router plugin."""
    provider: dict[str, Any] = {"data_collection": config.data_collection, "require_parameters": config.require_parameters}
    if config.max_price:
        provider["max_price"] = dict(config.max_price)
    body: dict[str, Any] = {"provider": provider}
    if config.fallbacks:
        body["models"] = [model_id, *config.fallbacks]
    if model_id == "openrouter/auto":
        auto = config.auto
        plugin: dict[str, Any] = {"id": "auto-router", "cost_tier": auto.cost_tier, "allowed_models": list(auto.allowed_models)}
        if auto.excluded_models:
            plugin["excluded_models"] = list(auto.excluded_models)
        body["plugins"] = [plugin]
        if auto.session == "run":
            body["session_id"] = session_id or uuid.uuid4().hex
    return body


def thinking_param(display: ThinkingDisplay) -> dict[str, Any] | None:
    """The request's ``thinking`` field (F9). Haiku, Sonnet and Opus 5.5 think adaptively by default and return the
    thinking empty; ``"summarized"`` asks for a readable summary, ``"omitted"`` sends nothing and keeps the default."""
    return {"type": "adaptive", "display": "summarized"} if display == "summarized" else None


def thinking_text(content: Iterable[Any]) -> str:
    """The thinking summary in a reply's content blocks, joined; empty when the display was omitted."""
    return "\n\n".join(t for b in content if getattr(b, "type", None) == "thinking" and (t := getattr(b, "thinking", "")))
