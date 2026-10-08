"""Textual app: chat pane plus the Guard-driven status bar."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from strands import Agent

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static
from textual.worker import Worker

from harness.agent import build_agent, stream_reply
from harness.config import HarnessConfig, load_config
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, run_preflight, system_gpu_limit
from harness.guard.watchdog import GuardState, Watchdog
from harness.status import StatusBar

# prompt -> stream of text deltas
Replier = Callable[[str], AsyncIterator[str]]


class HarnessApp(App[None]):
    CSS = """
    #chat { height: 1fr; padding: 0 1; }
    .user { color: $accent; margin-top: 1; }
    .assistant { margin-bottom: 1; }
    .note { color: $text-muted; }
    """
    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel generation", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        config: HarnessConfig,
        replier: Replier | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._client: OllamaClient | None = None
        self._agent: Agent | None = None
        self._watchdog = watchdog
        if replier is None:
            # Real mode: preflight (run on mount) pins num_ctx and builds the agent.
            self._client = OllamaClient(config.ollama.host, keep_alive=config.profile.keep_alive)
            self._watchdog = Watchdog(self._client, pinned=self._client.pinned, refresh_keep_alive=self._client.refresh_keep_alive)
            self._client.on_call(self._watchdog.record_call)
            replier = self._guarded_reply
        self._replier = replier
        self._worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Message (Enter to send, Ctrl-C cancels, Ctrl-Q quits)", id="prompt")
        yield StatusBar("")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        if self._watchdog is not None:
            self._watchdog.subscribe(self.query_one(StatusBar).set_state)
            self.run_worker(self._watchdog.run(), name="watchdog", group="guard")
        if self._client is not None:
            self.run_worker(self._boot(), name="preflight", group="guard")

    async def _note(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(text, classes="note", markup=False))
        chat.scroll_end(animate=False)

    async def _boot(self) -> None:
        """Preflight: pick a model and num_ctx that fit, load it, then hand the agent over."""
        assert self._client is not None
        profile = self.config.profile
        cands = [Candidate(m, profile.num_ctx) for m in (profile.model, *profile.fallbacks)]
        try:
            result = await run_preflight(self._client, cands, env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            for check in exc.checks:
                await self._note(f"preflight {check.severity.value}: {check.detail}")
            await self._note(f"preflight failed: {exc}")
            return
        for check in result.checks:
            if check.severity.value in ("warn", "fail"):
                await self._note(f"preflight {check.severity.value}: {check.detail}")
        await self._note(f"preflight ok: {result.model} at num_ctx {result.num_ctx} (budget {result.budget})")
        if self._watchdog is not None:
            self._watchdog.set_expected_size(result.model, result.weights)
        self._agent = build_agent(self._client, result.model, profile.system_prompt)

    def _guarded_reply(self, prompt: str) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("preflight has not finished (or failed); see the notes above")
        return stream_reply(self._agent, prompt)

    @property
    def guard_state(self) -> GuardState | None:
        return self._watchdog.state if self._watchdog is not None else None

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
