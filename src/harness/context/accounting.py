"""Per-call context accounting: which category of the Context window each token belongs to.

Works on the Ollama-format request actually sent, so the numbers describe what the model sees.
Counts are estimates (characters over an adaptive chars-per-token ratio); the ledger compares
them with Ollama's ``prompt_eval_count`` after every call.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MESSAGE_OVERHEAD_TOKENS = 4  # chat-template framing per message
INITIAL_CHARS_PER_TOKEN = 3.5


class Category(StrEnum):
    SYSTEM = "system"
    TOOLS = "tools"
    HISTORY = "history"
    TOOL_RESULTS = "tool results"
    TURN = "current turn"


@dataclass(frozen=True)
class Accounting:
    """Characters and message count per category for one request."""

    chars: Mapping[Category, int]
    messages: Mapping[Category, int]

    @property
    def total_chars(self) -> int:
        return sum(self.chars.values())

    @property
    def total_messages(self) -> int:
        return sum(self.messages.values())


def _message_chars(msg: Mapping[str, Any]) -> int:
    n = len(str(msg.get("content") or ""))
    for call in msg.get("tool_calls") or ():
        n += len(json.dumps(call.get("function", call), sort_keys=True))
    return n


def account(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None) -> Accounting:
    """Split a request into categories.

    The current turn is the last user message plus everything the agent added after it, except
    tool results, which keep their own category so a big read is visible as such.
    """
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=len(messages))
    chars = {c: 0 for c in Category}
    count = {c: 0 for c in Category}
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            cat = Category.SYSTEM
        elif msg.get("role") == "tool":
            cat = Category.TOOL_RESULTS
        elif i >= last_user:
            cat = Category.TURN
        else:
            cat = Category.HISTORY
        chars[cat] += _message_chars(msg)
        count[cat] += 1
    if tools:
        chars[Category.TOOLS] = len(json.dumps(list(tools), sort_keys=True))
    return Accounting(chars, count)


class TokenEstimator:
    """chars / ratio plus per-message framing, with the ratio learned from Ollama's own counts."""

    def __init__(self, chars_per_token: float = INITIAL_CHARS_PER_TOKEN) -> None:
        self.chars_per_token = chars_per_token
        self._chars = 0
        self._tokens = 0

    def tokens(self, chars: int, messages: int = 0) -> int:
        return math.ceil(chars / self.chars_per_token) + messages * MESSAGE_OVERHEAD_TOKENS

    def estimate(self, acct: Accounting) -> Mapping[Category, int]:
        return {c: self.tokens(acct.chars[c], acct.messages[c]) for c in Category}

    def observe(self, acct: Accounting, actual_tokens: int) -> None:
        """Fold in a call whose prompt was fully evaluated (not truncated), so ratio = cumulative chars / tokens."""
        body = actual_tokens - acct.total_messages * MESSAGE_OVERHEAD_TOKENS
        if acct.total_chars <= 0 or body <= 0:
            return
        self._chars += acct.total_chars
        self._tokens += body
        self.chars_per_token = self._chars / self._tokens
