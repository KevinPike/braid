"""Anthropic API usage and cost (ADR 0006, 0007): prices, context windows and the session's running meter.

Plain data and arithmetic over the counters in a response's ``usage``; nothing here imports ``anthropic``, so both the
chat model and the Workflow model record their calls the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Price:
    """USD per million tokens. A prompt (input + cache reads + cache writes) over ``long_over`` tokens bills at the long tier.

    ``cache_read`` is a cache-read price stated on its own (F10: Opus 5.5's $0.20 is 0.05x its input); None reads at
    0.1x the tier's input price. Cache writes cost 1.25x the tier's input price."""

    input: float
    output: float
    long_input: float
    long_output: float
    long_over: int = 100_000
    cache_read: float | None = None


# Sonnet and Opus 5.5 list no long-context tier, so their long prices are their standard ones.
PRICING: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.50, 2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00, 2.00, 10.00, cache_read=0.20),
    "claude-opus-5-5": Price(4.00, 20.00, 4.00, 20.00, cache_read=0.20),
}

# Tokens a request may use, input and output together.
CONTEXT_WINDOWS: dict[str, int] = {"claude-haiku-5-5": 1_000_000, "claude-sonnet-5-5": 1_000_000, "claude-opus-5-5": 1_000_000}


def require_pricing(model_id: str) -> Price:
    if model_id not in PRICING:
        raise ValueError(f"no pricing for {model_id!r}; add it to PRICING (known: {', '.join(PRICING)})")
    return PRICING[model_id]


@dataclass(frozen=True)
class ApiCall:
    """Usage and cost of one Anthropic API request, and the thinking summary its reply carried (F9)."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
    request_id: str | None
    thinking: str = ""  # empty when the display is "omitted", or the model did not think
    served_model: str = ""  # ``response.model``: another model than the one asked for after a server-side fallback (F10)
    thinking_on: bool | None = None  # set only when thinking is sampled: did this request leave thinking on?
    stop_reason: str = ""  # why the reply ended: end_turn, max_tokens (cut at the token cap), refusal, ...
    seconds: float | None = None  # wall time of the request, from sending it to having the whole reply

    @property
    def prompt_tokens(self) -> int:
        """Every token the prompt occupied: ``input_tokens`` excludes the ones read from or written to the cache."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


def api_call(
    model_id: str, input_tokens: int, output_tokens: int, cache_read: int | None, cache_write: int | None,
    request_id: str | None, *, thinking: str = "", served_model: str = "", cost_usd: float | None = None,
    thinking_on: bool | None = None, stop_reason: str = "", seconds: float | None = None,
) -> ApiCall:
    """Price one request's usage counters, at the prices of the model that served it when that one is priced. A
    ``cost_usd`` the API reported itself (OpenRouter, ADR 0009) is used as it is."""
    if cost_usd is not None:
        return ApiCall(input_tokens, output_tokens, cache_read or 0, cache_write or 0, cost_usd, request_id, thinking, served_model, thinking_on, stop_reason, seconds)
    price = PRICING.get(served_model) or require_pricing(model_id)
    read, write = cache_read or 0, cache_write or 0
    long = input_tokens + read + write > price.long_over
    price_in, price_out = (price.long_input, price.long_output) if long else (price.input, price.output)
    price_read = price.cache_read if price.cache_read is not None else price_in * 0.1
    cost = (input_tokens * price_in + read * price_read + write * price_in * 1.25 + output_tokens * price_out) / 1_000_000
    return ApiCall(input_tokens, output_tokens, read, write, cost, request_id, thinking, served_model, thinking_on, stop_reason, seconds)


@dataclass(frozen=True)
class UsageState:
    """What the status bar shows for an API-backed session."""

    model: str
    context_window: int
    calls: int = 0
    input_tokens: int = 0  # every prompt token, cached or not, summed over calls
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    last_prompt_tokens: int = 0  # the newest call's prompt plus its reply: what the next call starts from
    in_flight: bool = False


class UsageMeter:
    """The session's API calls: each request body as sent, then its priced usage. Listeners are plain functions."""

    def __init__(self, model: str, context_window: int) -> None:
        self.state = UsageState(model, context_window)
        self.calls: list[ApiCall] = []
        self._on_request: list[Callable[[str, Mapping[str, Any]], None]] = []
        self._on_call: list[Callable[[str, ApiCall], None]] = []
        self._on_state: list[Callable[[UsageState], None]] = []

    def on_request(self, listener: Callable[[str, Mapping[str, Any]], None]) -> None:
        self._on_request.append(listener)

    def on_call(self, listener: Callable[[str, ApiCall], None]) -> None:
        self._on_call.append(listener)

    def subscribe(self, listener: Callable[[UsageState], None]) -> None:
        self._on_state.append(listener)

    @property
    def cost_usd(self) -> float:
        return self.state.cost_usd

    def set_model(self, model: str, context_window: int) -> None:
        self._publish(UsageState(**{**self.state.__dict__, "model": model, "context_window": context_window}))

    def request(self, model: str, body: Mapping[str, Any]) -> None:
        for listener in self._on_request:
            listener(model, body)
        self._publish(UsageState(**{**self.state.__dict__, "in_flight": True}))

    def record(self, model: str, call: ApiCall, *, main: bool = True) -> None:
        """``main=False`` for a helper call (the summarizer): it costs money but says nothing about the context used."""
        self.calls.append(call)
        s = self.state
        self._publish(UsageState(
            s.model, s.context_window, s.calls + 1, s.input_tokens + call.prompt_tokens, s.output_tokens + call.output_tokens,
            s.cache_read_input_tokens + call.cache_read_input_tokens, s.cost_usd + call.cost_usd,
            call.prompt_tokens + call.output_tokens if main else s.last_prompt_tokens, False,
        ))
        for listener in self._on_call:
            listener(model, call)

    def abandon(self) -> None:
        """A request that never completed (cancelled or failed): no longer in flight."""
        self._publish(UsageState(**{**self.state.__dict__, "in_flight": False}))

    def _publish(self, state: UsageState) -> None:
        self.state = state
        for listener in self._on_state:
            listener(state)


def tokens(n: int) -> str:
    """``950``, ``12.3k``, ``1M``."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.2f}M".replace(".00M", "M")


def render_usage(state: UsageState) -> str:
    share = state.last_prompt_tokens / state.context_window if state.context_window else 0.0
    parts = [
        state.model,
        f"ctx {tokens(state.last_prompt_tokens)}/{tokens(state.context_window)} ({share:.0%})",
        f"in {tokens(state.input_tokens)} · out {tokens(state.output_tokens)} · cache {tokens(state.cache_read_input_tokens)}",
        f"${state.cost_usd:.4f}",
    ]
    if state.in_flight:
        parts.append("…")
    return "  ·  ".join(parts)
