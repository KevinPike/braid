"""Textual app: chat pane plus the Guard-driven status bar."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from strands import Agent
from strands.types.content import Message

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Input, Static
from textual.worker import Worker

from harness.agent import build_agent, stream_reply
from harness.chat import AlertBanner, ContextPanel, Reply
from harness.config import HarnessConfig, load_config
from harness.context.compaction import CompactionPolicy, Compactor
from harness.context.ledger import Ledger
from harness.context.strategies import build_conversation_manager, build_offloader, make_summarizer
from harness.context.trimlog import TrimEntry, TrimLog
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, run_preflight, system_gpu_limit
from harness.guard.watchdog import GuardState, Level, Watchdog
from harness.prompts import PromptHistory, PromptQueue
from harness.status import StatusBar

class PromptInput(Input):
    """Input whose Up/Down recall earlier prompts."""

    def __init__(self, history: PromptHistory, placeholder: str = "", id: str | None = None) -> None:
        super().__init__(placeholder=placeholder, id=id)
        self._history = history

    async def _on_key(self, event: Key) -> None:
        if event.key in ("up", "down"):
            text = self._history.previous(self.value) if event.key == "up" else self._history.next()
            event.stop()
            event.prevent_default()
            if text is not None:
                self.value = text
                self.cursor_position = len(text)
            return
        await super()._on_key(event)


# prompt -> stream of text deltas
Replier = Callable[[str], AsyncIterator[str]]


@dataclass(frozen=True)
class Command:
    """A slash command. Immediate ones run on submit; the rest wait their turn in the prompt queue."""

    usage: str
    immediate: bool


COMMANDS: dict[str, Command] = {
    "exit": Command("/exit", immediate=True),
    "model": Command("/model [name]", immediate=False),  # swaps the model, so it must not interrupt queued prompts
    "compact": Command("/compact", immediate=False),  # rewrites history, so it waits for the running turn
    "trims": Command("/trims", immediate=True),  # read-only
}


def parse_command(text: str) -> tuple[str, str] | None:
    """``/name args`` -> (name, args); None for ordinary prompts."""
    if not text.startswith("/"):
        return None
    name, _, args = text[1:].partition(" ")
    return name, args.strip()


class HarnessApp(App[None]):
    CSS = """
    #chat { height: 1fr; padding: 0 1; }
    .user { color: $accent; margin-top: 1; }
    .assistant { margin-bottom: 1; }
    .note { color: $text-muted; }
    """
    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel generation", priority=True),
        Binding("escape", "clear_queue", "Clear queue"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        config: HarnessConfig,
        replier: Replier | None = None,
        watchdog: Watchdog | None = None,
        trimlog: TrimLog | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.trimlog = trimlog or TrimLog(config.paths.trim_log_file if replier is None else ":memory:")
        self.compactor = compactor
        self.fake_messages: list[Message] = []  # history stand-in when a test supplies a replier instead of an agent
        self._client: OllamaClient | None = None
        self._agent: Agent | None = None
        self._model: str | None = None
        self._watchdog = watchdog
        if replier is None:
            # Real mode: preflight (run on mount) pins num_ctx and builds the agent.
            self._client = OllamaClient(config.ollama.host, keep_alive=config.profile.keep_alive)
            self._watchdog = Watchdog(self._client, pinned=self._client.pinned, refresh_keep_alive=self._client.refresh_keep_alive)
            self._client.on_call(lambda model, metrics, est: self._watchdog.record_call(model, metrics, est) if self._watchdog else None)
            replier = self._guarded_reply
        self._replier = replier
        self._worker: Worker[None] | None = None
        self._busy = False
        self._run_id = 0
        self.history = PromptHistory()
        self.queue = PromptQueue()
        self.ledger = Ledger(self._client.pinned if self._client is not None else lambda _m: None)

    def compose(self) -> ComposeResult:
        yield StatusBar("")
        yield VerticalScroll(id="chat")
        yield PromptInput(self.history, placeholder="Message (Enter sends, Ctrl-C cancels, Esc clears queue)", id="prompt")
        yield ContextPanel("", markup=True)
        yield AlertBanner("", markup=False)

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self.trimlog.subscribe(self._on_trim)
        self.ledger.subscribe(self.query_one(ContextPanel).set_record)
        self.query_one(ContextPanel).set_record(None)
        if self._watchdog is not None:
            self._watchdog.subscribe(self.query_one(StatusBar).set_state)
            self._watchdog.subscribe(self.query_one(AlertBanner).set_state)
            self._watchdog.subscribe(lambda _state: self._pump())
            self.run_worker(self._watchdog.run(), name="watchdog", group="guard")
        if self._client is not None:
            self.run_worker(self._boot(), name="preflight", group="guard")

    def _on_trim(self, entry: TrimEntry) -> None:
        self.call_later(self._note, f"trim {entry.line()}")

    async def _note(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(text, classes="note", markup=False))
        chat.scroll_end(animate=False)

    async def _boot(self) -> None:
        """Preflight: pick a model and num_ctx that fit, load it, then hand the agent over."""
        assert self._client is not None
        profile = self.config.profile
        cands = [Candidate(m, profile.num_ctx) for m in (profile.model, *profile.fallbacks)]
        await self._note(f"preflight: loading {cands[0].model} at num_ctx {cands[0].num_ctx} (this can take a while)")
        status = self.query_one(StatusBar)
        status.show_loading(cands[0].model)
        try:
            result = await run_preflight(self._client, cands, env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            status.show_loading(None)
            for check in exc.checks:
                await self._note(f"preflight {check.severity.value}: {check.detail}")
            await self._note(f"preflight failed: {exc}")
            return
        for check in result.checks:
            if check.severity.value in ("warn", "fail"):
                await self._note(f"preflight {check.severity.value}: {check.detail}")
        status.show_loading(None)
        await self._note(f"preflight ok: {result.model} at num_ctx {result.num_ctx} (budget {result.budget})")
        if self._watchdog is not None:
            self._watchdog.set_expected_size(result.model, result.weights)
        self._agent = self._make_agent(result.model)
        self._model = result.model
        await self._build_compactor(result.model)
        self._pump()

    def _make_agent(self, model: str) -> Agent:
        """An agent whose history is managed by the configured strategy (never Strands' silent default)."""
        assert self._client is not None
        cfg = self.config.context
        est = self.ledger.estimator
        offloader = build_offloader(cfg, self.config.paths, self.trimlog, est)
        return build_agent(
            self._client, model, self.config.profile.system_prompt, self.ledger,
            conversation_manager=build_conversation_manager(cfg, self.trimlog, est),
            plugins=[offloader] if offloader is not None else None,
        )

    async def _build_compactor(self, driver: str) -> None:
        """The custom strategy: summarize with the summarizer model, or the daily driver when it is not pulled."""
        assert self._client is not None
        cfg = self.config.context
        if cfg.strategy != "summarize":
            self.compactor = None
            return
        pulled = {t.name for t in await self._client.tags()}
        helper = cfg.summarizer_model != driver and cfg.summarizer_model in pulled
        model = cfg.summarizer_model if helper else driver
        num_ctx = cfg.summarizer_num_ctx if helper else (self._client.pinned(driver) or self.config.profile.num_ctx)
        if helper and self._watchdog is not None:
            self._watchdog.mark_helper(model)
        await self._note(f"compaction: summarize with {model} at {int(cfg.ceiling * 100)}% of num_ctx"
                         + ("" if helper else f" ({cfg.summarizer_model} not pulled, using the daily driver)"))
        client = self._client
        self.compactor = Compactor(
            trimlog=self.trimlog, ledger=self.ledger, num_ctx=lambda: client.pinned(driver) or 0,
            summarize=make_summarizer(self._client, model, num_ctx, self.ledger.estimator, unload_after=helper),
            policy=CompactionPolicy(cfg.ceiling, cfg.target, cfg.preserve_recent, cfg.pin_first),
        )

    @property
    def _messages(self) -> list[Message]:
        return self._agent.messages if self._agent is not None else self.fake_messages

    async def _compact_after_turn(self) -> None:
        """Runs inside the generation's worker, so the queue waits for it (compaction never overlaps a turn)."""
        if self.compactor is None:
            return
        state = self.guard_state
        if state is not None and any(a.kind == "truncation" for a in state.alerts):
            self.compactor.force("Truncation detected: Ollama dropped prompt tokens")
        reason = self.compactor.should_compact()
        if reason is None:
            return
        await self._note("compacting history…")
        self.compactor.pending_reason = reason
        await self.compactor.after_turn(self._messages)

    def _guarded_reply(self, prompt: str) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("preflight has not finished (or failed); see the notes above")
        return stream_reply(self._agent, prompt)

    @property
    def guard_state(self) -> GuardState | None:
        return self._watchdog.state if self._watchdog is not None else None

    @property
    def generating(self) -> bool:
        return self._busy

    @property
    def held_by_alert(self) -> bool:
        state = self.guard_state
        return state is not None and state.level == Level.RED

    def _refresh_queue_label(self) -> None:
        n = len(self.queue)
        if self.queue.paused:
            label = f"paused · {n} queued · Enter resumes, Esc clears"
        elif n and self.held_by_alert:
            label = f"{n} queued · held by alert"
        elif n:
            label = f"{n} queued"
        else:
            label = ""
        for prompt in self.query(PromptInput):  # empty while the app is shutting down
            prompt.border_title = label

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            if self.queue.paused:
                self.queue.resume()
                self._pump()
            self._refresh_queue_label()
            return
        self.history.push(text)
        parsed = parse_command(text)
        if parsed is not None:
            name, _ = parsed
            command = COMMANDS.get(name)
            if command is None:
                await self._note(f"unknown command: /{name} (commands: {', '.join(c.usage for c in COMMANDS.values())})")
                return
            if command.immediate:
                await self._run_command(text)
                return
        self.queue.put(text)
        self._pump()
        self._refresh_queue_label()

    async def _run_command(self, text: str) -> None:
        parsed = parse_command(text)
        assert parsed is not None
        name, args = parsed
        if name == "exit":
            self.exit()
        elif name == "model":
            await self._model_command(args)
        elif name == "compact":
            await self._compact_command()
        elif name == "trims":
            await self._trims_command()

    async def _compact_command(self) -> None:
        if self.compactor is None:
            await self._note("/compact: no compaction strategy active (set [context] strategy = \"summarize\")")
            return
        await self._note("compacting history…")
        entry = await self.compactor.compact(self._messages, "manual /compact")
        if entry is None:
            await self._note("nothing to compact yet")

    async def _trims_command(self) -> None:
        entries = self.trimlog.entries(limit=20)
        if not entries:
            await self._note("trim log is empty")
            return
        await self._note("trim log (most recent last):\n" + "\n".join(e.line() for e in entries))

    async def _model_command(self, name: str) -> None:
        if self._client is None:
            await self._note("/model: no model backend in this session")
            return
        if not name:
            current = self._model or "none"
            tags = ", ".join(t.name for t in await self._client.tags())
            await self._note(f"current model: {current}; available: {tags}")
            return
        await self._switch_model(name)

    async def _switch_model(self, name: str) -> None:
        """Unload the old model, then preflight and load the new one at the profile's num_ctx."""
        assert self._client is not None
        if name == self._model:
            await self._note(f"already on {name}")
            return
        profile = self.config.profile
        old = self._model
        await self._note(f"switching to {name} at num_ctx {profile.num_ctx} (unloading {old or 'nothing'} first; this can take a while)")
        status = self.query_one(StatusBar)
        status.show_loading(name)
        if old is not None:
            await self._client.unload(old)
        try:
            result = await run_preflight(self._client, [Candidate(name, profile.num_ctx)], env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            status.show_loading(None)
            await self._note(f"switch failed: {exc}")
            if old is not None:
                await self._note(f"staying on {old}; it reloads on the next prompt")
            return
        status.show_loading(None)
        for check in result.checks:
            if check.severity.value in ("warn", "fail"):
                await self._note(f"preflight {check.severity.value}: {check.detail}")
        if self._watchdog is not None:
            self._watchdog.set_expected_size(result.model, result.weights)
        previous = self._agent
        self._agent = self._make_agent(result.model)
        if previous is not None:
            self._agent.messages.extend(previous.messages)  # the conversation carries over
        self._model = result.model
        await self._build_compactor(result.model)
        await self._note(f"now on {result.model} at num_ctx {result.num_ctx} (budget {result.budget}); conversation kept")

    def _pump(self) -> None:
        """Start the next queued prompt if idle, not paused and no red Alert is active."""
        if self._busy or self.held_by_alert or (self._client is not None and self._agent is None):
            self._refresh_queue_label()
            return
        prompt = self.queue.pop()
        if prompt is None:
            self._refresh_queue_label()
            return
        self._busy = True
        self._run_id += 1
        self._worker = self.run_worker(self._generate(prompt, self._run_id), name="generate")
        self._refresh_queue_label()

    async def _generate(self, prompt: str, run_id: int) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        if parse_command(prompt) is not None:  # a queued slash command
            try:
                await self._run_command(prompt)
            except Exception as exc:
                await self._note(f"error: {exc}")
            finally:
                if run_id == self._run_id:
                    self._busy = False
                    self._pump()
            return
        reply = Reply()
        try:
            await chat.mount(Static(f"you: {prompt}", classes="user", markup=False), reply)
            chat.scroll_end(animate=False)
            try:
                async for delta in self._replier(prompt):
                    reply.feed(delta)
                    chat.scroll_end(animate=False)
            except Exception as exc:  # surface backend errors in the pane, don't crash the UI
                reply.add_class("note")
                reply.text += f"\n[error: {exc}]"
            finally:
                await reply.finish()
            try:
                await self._compact_after_turn()
            except Exception as exc:
                await self._note(f"compaction error: {exc}")
        finally:
            if run_id == self._run_id:  # a cancelled run may be unwound after the app has moved on
                self._busy = False
                self._pump()

    def action_cancel(self) -> None:
        """Cancel the generation and pause the queue so the next prompt never starts by itself."""
        if self.generating and self._worker is not None:
            self.queue.pause()
            self._worker.cancel()
            self._busy = False  # a worker cancelled before it starts never runs its finally
            self.query_one("#chat", VerticalScroll).mount(Static("[cancelled]", classes="note", markup=False))
            self._refresh_queue_label()

    def action_clear_queue(self) -> None:
        self.queue.clear()
        self._refresh_queue_label()


def main(argv: list[str] | None = None) -> None:
    import sys

    args = sys.argv[1:] if argv is None else argv
    config = load_config(Path(args[0]) if args else None)
    HarnessApp(config).run()
