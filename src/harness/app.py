"""Textual app: chat pane plus an (empty, for now) status bar."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static
from textual.worker import Worker

from harness.agent import build_agent, stream_reply
from harness.config import HarnessConfig, load_config

# prompt -> stream of text deltas
Replier = Callable[[str], AsyncIterator[str]]


class StatusBar(Static):
    """Placeholder; M1's guard fills it in."""


class HarnessApp(App[None]):
    CSS = """
    #chat { height: 1fr; padding: 0 1; }
    .user { color: $accent; margin-top: 1; }
    .assistant { margin-bottom: 1; }
    .note { color: $text-muted; }
    StatusBar { height: 1; background: $panel; padding: 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel generation", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config: HarnessConfig, replier: Replier | None = None) -> None:
        super().__init__()
        self.config = config
        if replier is None:
            agent = build_agent(config)
            replier = lambda prompt: stream_reply(agent, prompt)  # noqa: E731
        self._replier = replier
        self._worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Message (Enter to send, Ctrl-C cancels, Ctrl-Q quits)", id="prompt")
        yield StatusBar("")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @property
    def generating(self) -> bool:
        return self._worker is not None and self._worker.is_running

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.generating:
            return
        event.input.value = ""
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(f"you: {text}", classes="user", markup=False))
        reply = Static("", classes="assistant", markup=False)
        await chat.mount(reply)
        chat.scroll_end(animate=False)
        self._worker = self._generate(text, reply)

    @work(exclusive=True)
    async def _generate(self, prompt: str, reply: Static) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        buf = ""
        try:
            async for delta in self._replier(prompt):
                buf += delta
                reply.update(buf)
                chat.scroll_end(animate=False)
        except Exception as exc:  # surface backend errors in the pane, don't crash the UI
            reply.update(f"{buf}\n[error: {exc}]")
            reply.add_class("note")

    def action_cancel(self) -> None:
        if self.generating and self._worker is not None:
            self._worker.cancel()
            self.query_one("#chat", VerticalScroll).mount(Static("[cancelled]", classes="note", markup=False))


def main(argv: list[str] | None = None) -> None:
    import sys

    args = sys.argv[1:] if argv is None else argv
    config = load_config(Path(args[0]) if args else None)
    HarnessApp(config).run()
