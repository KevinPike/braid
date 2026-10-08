"""Prompt history (Up/Down recall) and the prompt queue; both pure so the TUI stays thin."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable


class PromptHistory:
    """Up/Down recall. ``entries`` (oldest first) seeds it from a persisted history."""

    def __init__(self, entries: Iterable[str] = ()) -> None:
        self._entries: list[str] = []
        for text in entries:
            self.push(text)
        self._cursor: int | None = None  # None = not browsing; the input holds the draft
        self._draft = ""

    def push(self, text: str) -> bool:
        """Record ``text``; False for empty or duplicate-consecutive prompts (the caller persists only on True)."""
        text = text.strip()
        self._cursor = None
        if text and (not self._entries or self._entries[-1] != text):
            self._entries.append(text)
            return True
        return False

    def previous(self, current: str) -> str | None:
        """Step back (newest first); the unsent draft is kept for ``next``. None at the oldest entry."""
        if not self._entries:
            return None
        if self._cursor is None:
            self._draft = current
            self._cursor = len(self._entries)
        if self._cursor == 0:
            return None
        self._cursor -= 1
        return self._entries[self._cursor]

    def next(self) -> str | None:
        """Step forward; past the newest entry returns the draft. None when not browsing."""
        if self._cursor is None:
            return None
        self._cursor += 1
        if self._cursor >= len(self._entries):
            self._cursor = None
            return self._draft
        return self._entries[self._cursor]


class PromptQueue:
    """FIFO of prompts waiting for the current generation. Ctrl-C pauses it; resume or clear is explicit."""

    def __init__(self) -> None:
        self._items: deque[str] = deque()
        self.paused = False
        self._listeners: list[Callable[[tuple[str, ...]], None]] = []

    def subscribe(self, listener: Callable[[tuple[str, ...]], None]) -> None:
        """Called with the pending prompts after every put, pop, clear and restore (the session store persists them)."""
        self._listeners.append(listener)

    def _changed(self) -> None:
        for listener in self._listeners:
            listener(self.pending)

    def restore(self, prompts: Iterable[str], *, paused: bool = True) -> None:
        """Put back prompts queued before a restart, ahead of anything queued since, paused so none auto-runs."""
        self._items.extendleft(reversed(list(prompts)))
        self.paused = paused or self.paused
        self._changed()

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._items)

    def put(self, prompt: str) -> None:
        self._items.append(prompt)
        self._changed()

    def pop(self) -> str | None:
        """Next prompt, or None if empty or paused."""
        if self.paused or not self._items:
            return None
        prompt = self._items.popleft()
        self._changed()
        return prompt

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        self.paused = False
        self._changed()
        return n
