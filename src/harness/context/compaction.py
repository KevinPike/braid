"""Compaction: keep the next call under the Context window ceiling by summarizing old history.

Runs between turns only (the app awaits ``after_turn`` before it releases the prompt queue), and
every step is written to the Trim log. Summarization goes through the guard client like every
other request; Strands' own summarizing manager is sync and runs its own event loop, which
cannot share our httpx client, so only its async helpers are reused here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from strands.agent.conversation_manager.compression.context_compression import (
    adjust_split_point_for_tool_pairs,
)
from strands.agent.conversation_manager.compression.pin_message import apply_pin_first, partition_pinned
from strands.types.content import Message
from strands.types.exceptions import ContextWindowOverflowException

from harness.context.accounting import TokenEstimator
from harness.context.ledger import Ledger
from harness.context.trimlog import TrimEntry, TrimLog

log = logging.getLogger(__name__)

SUMMARY_HEADER = "[Summary of the earlier conversation]\n"
MIN_TURN_GROWTH = 512  # tokens a turn is assumed to add before there is any history to measure
SUMMARY_PROMPT = (
    "You compress a conversation so the assistant can continue it. Write a dense third-person "
    "summary as bullet points. Keep every concrete fact exactly: names, numbers, dates, file paths, "
    "identifiers, decisions, user preferences and open questions. If an earlier summary is included, "
    "merge it in. Do not answer the conversation, address the user or add commentary."
)

Summarizer = Callable[[Sequence[Message]], Awaitable[str]]
# (tokens used, num_ctx, per-turn growth, turns so far) -> probability that history should be compacted now
Advisor = Callable[[int, int, int, int], Awaitable[float]]
ADVICE_THRESHOLD = 0.5


@dataclass(frozen=True)
class CompactionPolicy:
    ceiling: float = 0.9  # no call may exceed this share of num_ctx; compact before one would
    target: float = 0.5  # share of num_ctx to get back under
    preserve_recent: int = 6  # newest messages kept verbatim
    pin_first: int = 0  # oldest messages kept verbatim, never summarized


def message_text(message: Message) -> str:
    """Plain-text rendering of a Strands message, used for token estimates and the summarizer's transcript."""
    parts: list[str] = []
    for block in message["content"]:
        if "text" in block:
            parts.append(block["text"])
        elif "toolUse" in block:
            use = block["toolUse"]
            parts.append(f"[tool call {use['name']}({json.dumps(use.get('input', {}), sort_keys=True)})]")
        elif "toolResult" in block:
            body = " ".join(str(c.get("text", c.get("json", ""))) for c in block["toolResult"]["content"])
            parts.append(f"[tool result: {body}]")
    return "\n".join(parts)


def transcript(messages: Sequence[Message]) -> str:
    return "\n\n".join(f"{m['role']}: {message_text(m)}" for m in messages)


class Compactor:
    def __init__(
        self,
        *,
        trimlog: TrimLog,
        ledger: Ledger,
        summarize: Summarizer,
        num_ctx: Callable[[], int],
        policy: CompactionPolicy = CompactionPolicy(),
        strategy: str = "summarize",
        advisor: Advisor | None = None,
        advise_from: float = 0.8,
    ) -> None:
        self.trimlog = trimlog
        self.ledger = ledger
        self._summarize = summarize
        self._num_ctx = num_ctx
        self.policy = policy
        self.strategy = strategy
        self.pending_reason: str | None = None
        self.advisor = advisor
        self.advise_from = advise_from

    def estimate(self, messages: Sequence[Message]) -> int:
        est: TokenEstimator = self.ledger.estimator
        return sum(est.tokens(len(message_text(m)), 1) for m in messages)

    def used(self) -> int:
        """Tokens the last call's prompt plus its reply occupy: the floor for the next call."""
        for rec in reversed(self.ledger.records):
            if rec.prompt_eval_count is not None:
                return rec.prompt_eval_count + (rec.eval_count or 0)
        return 0

    def turn_growth(self) -> int:
        """Largest per-turn growth seen so far, so the ceiling holds for a turn as big as the biggest before it."""
        measured = [r.prompt_eval_count for r in self.ledger.records if r.prompt_eval_count is not None]
        deltas = [b - a for a, b in zip(measured, measured[1:]) if b > a]
        return max([MIN_TURN_GROWTH, *deltas])

    def should_compact(self) -> str | None:
        """Why to compact now, or None. A pending forced reason (e.g. Truncation) wins."""
        if self.pending_reason is not None:
            return self.pending_reason
        num_ctx = self._num_ctx()
        if not num_ctx or not self.ledger.records:
            return None
        projected = self.used() + self.turn_growth()
        if projected >= self.policy.ceiling * num_ctx:
            return f"next call projected at {projected}/{num_ctx} tokens, over {self.policy.ceiling:.0%} of num_ctx"
        return None

    def force(self, reason: str) -> None:
        self.pending_reason = reason

    async def advice(self) -> str | None:
        """Ask the decision layer once the window is ``advise_from`` full; it can only bring compaction forward.

        The projection against the ceiling stays the hard rule, so a wrong or failed answer costs nothing.
        """
        num_ctx = self._num_ctx()
        used = self.used()
        if self.advisor is None or not num_ctx or used < self.advise_from * num_ctx:
            return None
        try:
            p = await self.advisor(used, num_ctx, self.turn_growth(), len(self.ledger.records))
        except Exception as exc:
            log.warning("compaction advice failed: %s", exc)
            return None
        return f"decision layer advised compaction (p={p:.2f})" if p >= ADVICE_THRESHOLD else None

    async def after_turn(self, messages: list[Message]) -> TrimEntry | None:
        """Compact ``messages`` in place if the trigger says so; returns the Trim log entry."""
        reason = self.should_compact() or await self.advice()
        if reason is None:
            return None
        self.pending_reason = None
        return await self.compact(messages, reason)

    async def compact(self, messages: list[Message], reason: str) -> TrimEntry | None:
        pol = self.policy
        if pol.pin_first:
            apply_pin_first(messages, pol.pin_first)
        before = self.estimate(messages)
        # Shrink the verbatim tail until the result fits the target; always summarize at least one message.
        target_tokens = int(pol.target * self._num_ctx()) if self._num_ctx() else 0
        keep = max(0, pol.preserve_recent)
        while True:
            split = self._split(messages, keep)
            if split is None:
                if keep == 0:
                    log.warning("compaction skipped: nothing can be summarized (%s)", reason)
                    return None
                keep -= 1
                continue
            protected, to_summarize = partition_pinned(messages, 0, split)
            if not to_summarize:
                if keep == 0:
                    return None
                keep -= 1
                continue
            tail = messages[split:]
            if keep == 0 or not target_tokens or self.estimate(protected + tail) < target_tokens:
                break
            keep -= 1
        try:
            summary_text = await self._summarize(to_summarize)
        except Exception as exc:
            log.warning("summarizer failed: %s", exc)
            self.trimlog.record(TrimEntry(self.strategy, reason, 0, before, before, f"summarizer failed: {exc}"))
            return None
        summary: Message = {"role": "user", "content": [{"text": SUMMARY_HEADER + summary_text.strip()}]}
        messages[:] = protected + [summary] + tail
        after = self.estimate(messages)
        return self.trimlog.record(TrimEntry(
            self.strategy, reason, len(to_summarize), before, after,
            f"kept {len(protected)} pinned + {len(tail)} recent",
        ))

    @staticmethod
    def _split(messages: list[Message], keep: int) -> int | None:
        """Index where the verbatim tail starts, not breaking a tool call from its result."""
        count = len(messages) - keep
        if count <= 0:
            return None
        try:
            count = adjust_split_point_for_tool_pairs(messages, count)
        except ContextWindowOverflowException:
            return None
        return count if count > 0 else None

