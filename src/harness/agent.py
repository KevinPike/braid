"""Strands Agent whose model streams through the guard client (ADR 0001)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import Any

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.types.streaming import StreamEvent

from harness.context.ledger import Ledger
from harness.guard.ollama_client import OllamaClient


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


def build_agent(client: OllamaClient, model_id: str, system_prompt: str, ledger: Ledger | None = None) -> Agent:
    # callback_handler=None: the TUI renders the stream, nothing prints to stdout.
    return Agent(model=GuardedOllamaModel(client, model_id, ledger), system_prompt=system_prompt, callback_handler=None)


async def stream_reply(agent: Agent, prompt: str) -> AsyncIterator[str]:
    """Yield text deltas for one turn."""
    async for event in agent.stream_async(prompt):
        delta = event.get("data")
        if isinstance(delta, str) and delta:
            yield delta
