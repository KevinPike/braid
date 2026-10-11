"""The chat agent's Strands models: Claude on the Anthropic API (ADR 0007), or Ollama through the guard client (ADR 0001)."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import anthropic
from strands import Agent
from strands.agent.conversation_manager import ConversationManager
from strands.hooks import HookCallback, HookProvider
from strands.models.anthropic import AnthropicModel
from strands.models.model import Model
from strands.plugins import Plugin
from strands.models.ollama import OllamaModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from harness.config import Effort
from harness.context.ledger import Ledger
from harness.guard.ollama_client import CallMetrics, OllamaClient
from harness.usage import CONTEXT_WINDOWS, UsageMeter, api_call, require_pricing

log = logging.getLogger(__name__)


class GuardedOllamaModel(OllamaModel):
    """Reuses Strands' request formatting but sends every request through ``OllamaClient``.

    That keeps ``num_ctx`` pinned and surfaces per-call metrics to the watchdog; Strands'
    own ollama client is never used.
    """

    def __init__(self, client: OllamaClient, model_id: str, ledger: Ledger | None = None, **model_config: Any) -> None:
        super().__init__(client.host, model_id=model_id, **model_config)
        self._guard = client
        self._ledger = ledger

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        request = self.format_request(messages, tool_specs, system_prompt)
        estimate: int | None = None
        if self._ledger is not None:
            theirs = await self.count_tokens(messages, tool_specs, system_prompt)
            estimate = self._ledger.before_call(request["model"], request["messages"], request["tools"], theirs)
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        tool_requested = False
        last = None
        async for chunk in self._guard.chat(
            request["model"], request["messages"], tools=request["tools"], options=request["options"], estimate=estimate
        ):
            last = chunk
            for call in chunk.tool_calls:
                fn = call["function"]
                shim = SimpleNamespace(function=SimpleNamespace(name=fn["name"], arguments=fn.get("arguments", {})))
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": shim})
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": shim})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool", "data": shim})
                tool_requested = True
            yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": chunk.text})

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "message_stop", "data": "tool_use" if tool_requested else "stop"})
        if last is not None and last.metrics is not None:
            m = last.metrics
            meta = SimpleNamespace(
                prompt_eval_count=m.prompt_eval_count, eval_count=m.eval_count, total_duration=m.total_duration_s * 1e9
            )
            yield self.format_chunk({"chunk_type": "metadata", "data": meta})
            if self._ledger is not None:
                self._ledger.after_call(m)


# Stream events the chat model passes on; the rest (``message_delta``, the SDK's own ``text``/``thinking`` helpers) are
# read from the final message instead.
_PASSED = {"content_block_start", "content_block_delta", "content_block_stop"}
_DELTAS = {"text_delta", "input_json_delta", "thinking_delta", "signature_delta"}
OVERFLOW_MESSAGES = AnthropicModel.OVERFLOW_MESSAGES


def ollama_view(request: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """An Anthropic request as the Ollama-shaped (messages, tools) the Ledger's accounting reads: tool results are
    ``tool`` messages, so the context panel splits an API call into the same categories as an Ollama one."""
    out: list[dict[str, Any]] = []
    for block in request.get("system") or ():
        out.append({"role": "system", "content": block["text"]})
    for message in request["messages"]:
        text: list[str] = []
        calls: list[dict[str, Any]] = []
        for block in message["content"]:
            kind = block.get("type")
            if kind == "text":
                text.append(block["text"])
            elif kind == "tool_use":
                calls.append({"function": {"name": block["name"], "arguments": block["input"]}})
            elif kind == "tool_result":
                body = " ".join(c.get("text", "") for c in block.get("content") or () if isinstance(c, Mapping))
                out.append({"role": "tool", "content": body})
        if text or calls:
            out.append({"role": message["role"], "content": "\n".join(text), **({"tool_calls": calls} if calls else {})})
    return out, list(request.get("tools") or ())


class AnthropicChatModel(AnthropicModel):
    """The chat agent on Claude through the Anthropic SDK 1.x (ADR 0007).

    Strands' own provider declares ``anthropic<1.0``, so only its pure formatting is reused here (Strands messages to
    API content blocks, API stream events to Strands ``StreamEvent``s); the request itself is built and sent here, on the
    1.x client handed in. It never sends sampling parameters, caches the system prompt and the history prefix, and
    records every call's priced usage on ``meter`` and in the Ledger (in place of Ollama's ``prompt_eval_count``).
    Cancelling the task that iterates ``stream`` closes the HTTP stream, which cancels the request.
    """

    def __init__(
        self, client: anthropic.AsyncAnthropic, model_id: str, *, max_tokens: int = 16000, effort: Effort | None = None,
        meter: UsageMeter | None = None, ledger: Ledger | None = None, context_window: int | None = None,
    ) -> None:
        require_pricing(model_id)
        # Not ``super().__init__``: that would build a client of its own. Only ``config`` is read by the formatters.
        self.config = AnthropicModel.AnthropicConfig(model_id=model_id, max_tokens=max_tokens)
        self.client = client
        self.effort = effort
        self.meter = meter
        self.ledger = ledger
        self.context_window = context_window or CONTEXT_WINDOWS.get(model_id, 200_000)

    @property
    def model_id(self) -> str:
        return self.config["model_id"]

    def request(self, messages: Messages, tool_specs: Sequence[ToolSpec] | None, system_prompt: str | None) -> dict[str, Any]:
        """The request body: tools, a cached system prompt, then the history, with automatic caching of its prefix."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.config["max_tokens"],
            "messages": self._format_request_messages(messages),
            "cache_control": {"type": "ephemeral"},  # caches up to the last block: the next call reads the history back
        }
        if system_prompt:
            body["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        if tool_specs:
            body["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]["json"]}
                for t in tool_specs
            ]
        if self.effort is not None:
            body["output_config"] = {"effort": self.effort}
        return body

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        body = self.request(messages, tool_specs, system_prompt)
        if self.ledger is not None:
            view, tools = ollama_view(body)
            theirs = await Model.count_tokens(self, messages, tool_specs, system_prompt)
            self.ledger.before_call(self.model_id, view, tools, theirs)
        if self.meter is not None:
            self.meter.request(self.model_id, body)
        done = False
        try:
            async with self.client.messages.stream(**body) as stream:
                yield self.format_chunk({"type": "message_start"})
                async for event in stream:
                    if event.type not in _PASSED:
                        continue
                    if event.type == "content_block_stop":
                        yield self.format_chunk({"type": "content_block_stop", "index": event.index})
                        continue
                    payload = event.model_dump()
                    if event.type == "content_block_delta" and payload["delta"]["type"] not in _DELTAS:
                        continue  # a delta type this harness does not keep (citations, new kinds)
                    yield self.format_chunk(payload)
                final = await stream.get_final_message()
                request_id = stream.request_id
            done = True
        except anthropic.RateLimitError as exc:
            raise ModelThrottledException(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            if any(m in str(exc).lower() for m in OVERFLOW_MESSAGES):
                raise ContextWindowOverflowException(str(exc)) from exc
            raise
        finally:
            if not done and self.meter is not None:
                self.meter.abandon()
        stop = final.stop_reason or "end_turn"
        if stop == "refusal":
            category = final.stop_details.category if final.stop_details else None
            note = f"[{self.model_id} declined this request (category: {category})]"
            index = len(final.content)
            yield self.format_chunk({"type": "content_block_start", "index": index, "content_block": {"type": "text"}})
            yield self.format_chunk({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": note}})
            yield self.format_chunk({"type": "content_block_stop", "index": index})
            stop = "end_turn"
        elif stop == "pause_turn":  # only server tools pause, and this agent declares none
            stop = "end_turn"
        yield self.format_chunk({"type": "message_stop", "message": {"stop_reason": stop}})
        u = final.usage
        call = api_call(
            self.model_id, u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens,
            request_id,
        )
        yield self.format_chunk({"type": "metadata", "usage": u.model_dump()})
        if self.ledger is not None:
            self.ledger.after_call(CallMetrics(call.prompt_tokens, call.output_tokens, 0.0, 0.0))
        if self.meter is not None:
            self.meter.record(self.model_id, call)


def build_agent(
    client: OllamaClient,
    model_id: str,
    system_prompt: str,
    ledger: Ledger | None = None,
    *,
    conversation_manager: ConversationManager | None = None,
    plugins: list[Plugin] | None = None,
    tools: list[Any] | None = None,
    hooks: list[HookProvider | HookCallback[Any]] | None = None,
) -> Agent:
    # callback_handler=None: the TUI renders the stream, nothing prints to stdout.
    return Agent(
        model=GuardedOllamaModel(client, model_id, ledger), system_prompt=system_prompt, callback_handler=None,
        conversation_manager=conversation_manager, plugins=plugins, tools=tools, hooks=hooks,
    )


def build_api_agent(
    model: AnthropicChatModel,
    system_prompt: str,
    *,
    conversation_manager: ConversationManager | None = None,
    plugins: list[Plugin] | None = None,
    tools: list[Any] | None = None,
    hooks: list[HookProvider | HookCallback[Any]] | None = None,
) -> Agent:
    return Agent(
        model=model, system_prompt=system_prompt, callback_handler=None,
        conversation_manager=conversation_manager, plugins=plugins, tools=tools, hooks=hooks,
    )


async def stream_reply(agent: Agent, prompt: str) -> AsyncIterator[str]:
    """Yield text deltas for one turn."""
    async for event in agent.stream_async(prompt):
        delta = event.get("data")
        if isinstance(delta, str) and delta:
            yield delta
