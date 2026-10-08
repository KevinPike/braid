"""Wiring for the M3 strategies: Strands managers and plugins that log to the Trim log, and the summarizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from strands.agent.conversation_manager import (
    ConversationManager,
    NullConversationManager,
    SlidingWindowConversationManager,
)
from strands.agent.conversation_manager.compression.context_compression import generate_summary
from strands.plugins import Plugin
from strands.types.content import Message
from strands.vended_plugins.context_offloader import ContextOffloader, FileStorage

from harness.agent import GuardedOllamaModel
from harness.config import ContextConfig, PathsConfig
from harness.context.accounting import TokenEstimator
from harness.context.compaction import SUMMARY_PROMPT, Summarizer, message_text, transcript
from harness.context.trimlog import TrimEntry, TrimLog
from harness.guard.ollama_client import OllamaClient

SUMMARIZER_SHARE = 0.6  # of the summarizer's num_ctx a chunk may fill; the rest is for the prompt and the summary


class LoggedSlidingWindow(SlidingWindowConversationManager):
    """Strands' sliding window, with every reduction written to the Trim log."""

    def __init__(self, window_size: int, trimlog: TrimLog, estimator: TokenEstimator, pin_first: int = 0) -> None:
        super().__init__(window_size=window_size, pin_first=pin_first or None)
        self._trimlog = trimlog
        self._estimator = estimator

    def _tokens(self, messages: Sequence[Message]) -> int:
        return sum(self._estimator.tokens(len(message_text(m)), 1) for m in messages)

    def reduce_context(self, agent: Any, e: Exception | None = None, **kwargs: Any) -> None:
        before_messages, before = len(agent.messages), self._tokens(agent.messages)
        super().reduce_context(agent, e, **kwargs)
        removed = before_messages - len(agent.messages)
        # Truncated tool results shrink tokens without removing messages, so log those too.
        after = self._tokens(agent.messages)
        if removed or after < before:
            self._trimlog.record(TrimEntry(
                "sliding_window", f"history over {self.window_size} messages", removed, before, after,
            ))


class LoggedFileStorage:
    """Legacy offloader storage over ``FileStorage`` that records each offload in the Trim log."""

    def __init__(self, directory: str, trimlog: TrimLog, estimator: TokenEstimator, preview_tokens: int) -> None:
        self._inner = FileStorage(directory)
        self._trimlog = trimlog
        self._estimator = estimator
        self._preview_tokens = preview_tokens

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        reference = await self._inner.store(key, content, content_type)
        before = self._estimator.tokens(len(content))
        self._trimlog.record(TrimEntry(
            "offload", f"tool result of {len(content)} bytes over the offload threshold", 0,
            before, min(before, self._preview_tokens), f"stored as {reference}",
        ))
        return reference

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        return await self._inner.retrieve(reference)


def build_offloader(cfg: ContextConfig, paths: PathsConfig, trimlog: TrimLog, estimator: TokenEstimator) -> Plugin | None:
    if cfg.offload_tokens <= 0:
        return None
    preview = max(1, cfg.offload_tokens // 4)
    storage = LoggedFileStorage(str(paths.offload_dir), trimlog, estimator, preview)
    return ContextOffloader(storage=storage, max_result_tokens=cfg.offload_tokens, preview_tokens=preview)


def build_conversation_manager(cfg: ContextConfig, trimlog: TrimLog, estimator: TokenEstimator) -> ConversationManager:
    """Strands' default manager would trim silently after 40 messages, so every strategy names its own."""
    if cfg.strategy == "sliding_window":
        return LoggedSlidingWindow(cfg.window_size, trimlog, estimator, cfg.pin_first)
    return NullConversationManager()


def chunk_messages(messages: Sequence[Message], budget_tokens: int, estimator: TokenEstimator) -> list[list[Message]]:
    """Greedy split so each chunk's transcript fits ``budget_tokens``; an oversized message gets a chunk of its own."""
    chunks: list[list[Message]] = []
    current: list[Message] = []
    used = 0
    for m in messages:
        cost = estimator.tokens(len(message_text(m)) + len(m["role"]) + 2, 1)
        if current and used + cost > budget_tokens:
            chunks.append(current)
            current, used = [], 0
        current.append(m)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def _clip(message: Message, max_chars: int) -> Message:
    text = message_text(message)
    if len(text) <= max_chars:
        return message
    return {"role": message["role"], "content": [{"text": text[: max_chars // 2] + "\n[...clipped...]\n" + text[-max_chars // 2 :]}]}


def make_summarizer(client: OllamaClient, model: str, num_ctx: int, estimator: TokenEstimator, *, unload_after: bool) -> Summarizer:
    """Summarize through the guard client at the summarizer's own pinned num_ctx.

    Long input is folded chunk by chunk (each chunk carries the summary so far), so the prompt never
    fills the summarizer's window, which would be Truncation.
    """
    async def summarize(messages: Sequence[Message]) -> str:
        if client.pinned(model) is None:
            client.pin(model, num_ctx)
        model_obj = GuardedOllamaModel(client, model)
        budget = int(num_ctx * SUMMARIZER_SHARE)
        max_chars = int(budget * estimator.chars_per_token)
        running = ""
        try:
            for chunk in chunk_messages([_clip(m, max_chars) for m in messages], budget, estimator):
                feed: list[Message] = list(chunk)
                if running:
                    feed.insert(0, {"role": "user", "content": [{"text": "Summary so far:\n" + running}]})
                reply = await generate_summary(feed, model_obj, SUMMARY_PROMPT)
                running = message_text(reply)
        finally:
            if unload_after:
                await client.unload(model)
        return running

    return summarize


__all__ = ["build_conversation_manager", "build_offloader", "make_summarizer", "transcript"]
