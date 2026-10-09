"""Textual app: chat pane plus the Guard-driven status bar."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent
from strands.hooks import MessageAddedEvent
from strands.types.content import Message

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Input, Static
from textual.worker import Worker

from harness.agent import build_agent, stream_reply
from harness.approval import ApprovalGate, ApprovalRequest
from harness.chat import AlertBanner, ApprovalPrompt, ContextPanel, Reply
from harness.config import HarnessConfig, load_config
from harness.context.compaction import CompactionPolicy, Compactor, message_text
from harness.context.ledger import Ledger
from harness.context.strategies import build_conversation_manager, build_offloader, make_summarizer
from harness.context.trimlog import TrimLog
from harness.decide.log import DecisionLog
from harness.decide.questions import check_policy, route, should_compact
from harness.decide.tev import TevClient
from harness.tools.builtin import build_shell, build_tools, build_write_file
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, PreflightResult, run_preflight, system_gpu_limit
from harness.guard.watchdog import GuardState, Level, Watchdog
from harness.hooks import (
    ApprovalResolved, CallRecorded, DecisionRecorded, GuardStateChanged, HarnessHooks, ModelCallCompleted,
    ModelRequestSent, QueueChanged, TrimRecorded,
)
from harness.inspector import Inspector
from harness.prompts import PromptHistory, PromptQueue
from harness.session.store import SessionInfo, SessionStore, settle
from harness.status import StatusBar
from harness.telemetry import setup_telemetry

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
    "tools": Command("/tools", immediate=True),
    "decisions": Command("/decisions", immediate=True),
    "sessions": Command("/sessions", immediate=True),  # read-only
    "resume": Command("/resume <id>", immediate=False),  # swaps the history, so it waits for the running turn
    "fork": Command("/fork [id]", immediate=False),
    "new": Command("/new", immediate=False),
    "inspect": Command("/inspect [n]", immediate=True),
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
        Binding("ctrl+t", "inspect", "Transcript inspector"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        config: HarnessConfig,
        replier: Replier | None = None,
        watchdog: Watchdog | None = None,
        trimlog: TrimLog | None = None,
        compactor: Compactor | None = None,
        decisions: DecisionLog | None = None,
        tev: TevClient | None = None,
        store: SessionStore | None = None,
        resume: str | None = None,
        continue_last: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hooks = HarnessHooks()  # every Guard, context, decision, session and approval event; M9 registers here
        self._telemetry = setup_telemetry(config.telemetry, self.hooks)  # None unless [telemetry] enabled
        self._approvals_waiting = 0
        self.trimlog = trimlog or TrimLog(config.paths.trim_log_file if replier is None else ":memory:")
        self.compactor = compactor
        self.decisions = decisions or DecisionLog(config.paths.decisions_file if replier is None else ":memory:")
        self.tev = tev
        self.store = store or SessionStore(config.paths.sessions_file if replier is None else ":memory:")
        self._startup_notes: list[str] = []
        self._resume_info = self._pick_session(resume, continue_last)
        info = self._resume_info or self.store.create(config.profile.model, config.profile.num_ctx)
        self.session_id = info.id
        self._preflight: PreflightResult | None = None
        self._open_calls: dict[str, int] = {}  # model -> the calls row awaiting its measured counts
        self.fake_messages: list[Message] = []  # history stand-in when a test supplies a replier instead of an agent
        self._client: OllamaClient | None = None
        self._agent: Agent | None = None
        self._model: str | None = None
        self._watchdog = watchdog
        if replier is None:
            # Real mode: preflight (run on mount) pins num_ctx and builds the agent.
            self._client = OllamaClient(config.ollama.host, keep_alive=config.profile.keep_alive)
            self._watchdog = Watchdog(self._client, pinned=self._client.pinned, refresh_keep_alive=self._client.refresh_keep_alive)
            replier = self._guarded_reply
        self._replier = replier
        self._worker: Worker[None] | None = None
        self._busy = False
        self._run_id = 0
        self.history = PromptHistory(p.text for p in self.store.prompts())  # global recall across sessions
        self.queue = PromptQueue()
        self.ledger = Ledger(self._client.pinned if self._client is not None else lambda _m: None)
        # Each component is subscribed to exactly once, by the adapter; everything else registers on self.hooks.
        if self._client is not None:
            self.hooks.attach_client(self._client)
        if self._watchdog is not None:
            self.hooks.attach_watchdog(self._watchdog)
        self.hooks.attach_queue(self.queue)
        self.hooks.attach_ledger(self.ledger)
        self.hooks.attach_trimlog(self.trimlog)
        self.hooks.attach_decisions(self.decisions)
        self.hooks.add_callback(QueueChanged, lambda e: self.store.save_queue(self.session_id, e.pending))
        self.hooks.add_callback(ModelRequestSent, self._on_request)
        self.hooks.add_callback(ModelCallCompleted, self._on_call_done)
        self.hooks.add_callback(TrimRecorded, self._on_trim)
        self.hooks.add_callback(DecisionRecorded, self._on_decision)
        self.hooks.add_callback(ApprovalResolved, self._on_approval_resolved)

    def compose(self) -> ComposeResult:
        yield StatusBar("")
        yield VerticalScroll(id="chat")
        yield ApprovalPrompt()
        yield PromptInput(self.history, placeholder="Message (Enter sends, Ctrl-C cancels, Esc clears queue)", id="prompt")
        yield ContextPanel("", markup=True)
        yield AlertBanner("", markup=False)

    def _pick_session(self, resume: str | None, continue_last: bool) -> SessionInfo | None:
        """The session to resume at startup, or None for a new one (with a note when the request could not be met)."""
        if resume is not None:
            info = self.store.resolve(resume)
            if info is None:
                self._startup_notes.append(f"no single session matches '{resume}'; starting a new one (/sessions lists them)")
            return info
        if continue_last:
            info = self.store.latest()
            if info is None:
                self._startup_notes.append("--continue: no earlier session; starting a new one")
            return info
        return None

    async def on_mount(self) -> None:
        self.query_one(Input).focus()
        panel, status, banner = self.query_one(ContextPanel), self.query_one(StatusBar), self.query_one(AlertBanner)
        self.hooks.add_callback(CallRecorded, lambda e: panel.set_record(e.record))
        panel.set_record(None)
        if self._watchdog is not None:
            self.hooks.add_callback(GuardStateChanged, lambda e: status.set_state(e.state))
            self.hooks.add_callback(GuardStateChanged, lambda e: banner.set_state(e.state))
            self.hooks.add_callback(GuardStateChanged, lambda _e: self._pump())
            self.run_worker(self._watchdog.run(), name="watchdog", group="guard")
        if self._client is not None:
            self.run_worker(self._boot(), name="preflight", group="guard")
        for note in self._startup_notes:
            await self._note(note)
        if self._client is None and self._resume_info is not None:  # real mode adopts it once the model is loaded
            await self._adopt(self._resume_info)

    def _on_trim(self, event: TrimRecorded) -> None:
        self.call_later(self._note, f"trim {event.entry.line()}")

    def _on_decision(self, event: DecisionRecorded) -> None:
        self.call_later(self._note, f"decide {event.decision.line()}")

    def _on_approval_resolved(self, event: ApprovalResolved) -> None:
        self.call_later(self._note, f"{'approved' if event.approved else 'denied'} {event.request.label()}")

    def on_unmount(self) -> None:
        if self._telemetry is not None:
            self._telemetry.shutdown()

    async def _note(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(text, classes="note", markup=False))
        chat.scroll_end(animate=False)

    async def _boot(self) -> None:
        """Preflight: pick a model and num_ctx that fit, load it, then hand the agent over."""
        assert self._client is not None
        profile = self.config.profile
        cands = [Candidate(m, profile.num_ctx) for m in (profile.model, *profile.fallbacks)]
        if self._resume_info is not None:  # the session's own model and num_ctx first; the profile is the fallback
            cands.insert(0, Candidate(self._resume_info.model, self._resume_info.num_ctx))
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
        self._preflight = result
        await self._build_helpers(result.model)
        if self._resume_info is not None:
            if (result.model, result.num_ctx) != (self._resume_info.model, self._resume_info.num_ctx):
                await self._note(f"session was on {self._resume_info.model} at num_ctx {self._resume_info.num_ctx}; "
                                 f"resuming on {result.model} at {result.num_ctx} instead")
            await self._adopt(self._resume_info)
        else:
            self._save_session()
        self._pump()

    def _make_agent(self, model: str) -> Agent:
        """An agent whose history is managed by the configured strategy (never Strands' silent default)."""
        assert self._client is not None
        cfg = self.config.context
        est = self.ledger.estimator
        offloader = build_offloader(cfg, self.config.paths, self.trimlog, est)
        gate = ApprovalGate(self.ask_approval, self._advise_policy, gated=self.config.tools.approve, hooks=self.hooks)
        agent = build_agent(
            self._client, model, self.config.profile.system_prompt, self.ledger,
            conversation_manager=build_conversation_manager(cfg, self.trimlog, est),
            plugins=[offloader] if offloader is not None else None,
            tools=self._tools(),
            hooks=[gate],  # the write and shell tools are only ever registered on an agent that has this gate
        )
        # Every added message is saved at once, so a kill mid-turn keeps what the turn had so far.
        agent.hooks.add_callback(MessageAddedEvent, lambda e: self.store.save_messages(self.session_id, e.agent.messages))
        return agent

    def _tools(self) -> list[Any]:
        cfg = self.config.tools
        if not cfg.enabled:
            return []
        tools: list[Any] = build_tools(cfg.root, cfg.notes_dir, max_read_chars=cfg.max_read_chars)
        if cfg.write:
            tools.append(build_write_file(cfg.root))
        if cfg.shell:
            tools.append(build_shell(cfg.root, network=cfg.shell_network))
        return tools

    @property
    def approval_pending(self) -> bool:
        return self._approvals_waiting > 0

    async def ask_approval(self, request: ApprovalRequest) -> bool:
        """The Approver: show the approval prompt and wait for y or n. While it is open the queue is held."""
        prompt = self.query_one(ApprovalPrompt)
        self._approvals_waiting += 1
        self._refresh_queue_label()
        try:
            return await prompt.ask(request)
        finally:
            self._approvals_waiting -= 1
            self._pump()

    async def _advise_policy(self, tool: str, tool_input: Mapping[str, Any]) -> float | None:
        """tev1's policy check, as advice for the approval prompt only (None when the decision layer is off)."""
        return await check_policy(self.tev, tool, tool_input) if self.tev is not None else None

    async def _build_helpers(self, driver: str) -> None:
        """Summarizer (compaction and oversized tev1 input) and the tev1 client, each only when its model is pulled."""
        assert self._client is not None
        client = self._client
        cfg = self.config.context
        pulled = {t.name for t in await client.tags()}
        helper = cfg.summarizer_model != driver and cfg.summarizer_model in pulled
        model = cfg.summarizer_model if helper else driver
        num_ctx = cfg.summarizer_num_ctx if helper else (client.pinned(driver) or self.config.profile.num_ctx)
        if helper and self._watchdog is not None:
            self._watchdog.mark_helper(model)
        summarizer = make_summarizer(client, model, num_ctx, self.ledger.estimator, unload_after=helper)

        self.tev = None
        dec = self.config.decide
        if dec.enabled and dec.model in pulled:
            async def shorten(text: str, budget: int) -> str:
                return await summarizer([{"role": "user", "content": [{"text": text}]}])

            if self._watchdog is not None:
                self._watchdog.mark_helper(dec.model)
            self.tev = TevClient(
                self.config.ollama.host, dec.model, log=self.decisions, summarize=shorten,
                max_input_tokens=dec.max_input_tokens, keep_alive=dec.keep_alive,
            )
        elif dec.enabled:
            await self._note(f"decision layer off: {dec.model} is not pulled")

        self.compactor = None
        if cfg.strategy != "summarize":
            return
        await self._note(f"compaction: summarize with {model} at {int(cfg.ceiling * 100)}% of num_ctx"
                         + ("" if helper else f" ({cfg.summarizer_model} not pulled, using the daily driver)"))
        self.compactor = Compactor(
            trimlog=self.trimlog, ledger=self.ledger, num_ctx=lambda: client.pinned(driver) or 0,
            summarize=summarizer, policy=CompactionPolicy(cfg.ceiling, cfg.target, cfg.preserve_recent, cfg.pin_first),
            advisor=self._advise_compaction if self.tev is not None else None, advise_from=dec.compaction_advice_from,
        )

    async def _advise_compaction(self, used: int, num_ctx: int, growth: int, turns: int) -> float:
        assert self.tev is not None
        return await should_compact(self.tev, used=used, num_ctx=num_ctx, growth=growth, turns=turns)

    async def _route(self, prompt: str) -> None:
        """Advisory: the router's answer is logged (and shown); nothing acts on it until model routing in M10."""
        assert self.tev is not None
        try:
            await route(self.tev, prompt)
        except Exception:
            pass  # TevClient.ask already put the failure in the Decision log

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
        reason = self.compactor.should_compact() or await self.compactor.advice()
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
        elif n and self.approval_pending:
            label = f"{n} queued · held by approval"
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
        self.store.add_prompt(self.session_id, text)
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
        elif name == "tools":
            await self._tools_command()
        elif name == "decisions":
            await self._decisions_command()
        elif name == "sessions":
            await self._sessions_command()
        elif name == "resume":
            await self._resume_command(args)
        elif name == "fork":
            await self._fork_command(args)
        elif name == "new":
            await self._switch_session(self.store.create(self._model_name, self._num_ctx, self._calibration()))
        elif name == "inspect":
            await self._inspect_command(args)

    async def _compact_command(self) -> None:
        if self.compactor is None:
            await self._note("/compact: no compaction strategy active (set [context] strategy = \"summarize\")")
            return
        await self._note("compacting history…")
        entry = await self.compactor.compact(self._messages, "manual /compact")
        if entry is None:
            await self._note("nothing to compact yet")
        else:
            self._save_session()

    async def _tools_command(self) -> None:
        if self._agent is None:
            await self._note("/tools: no agent yet")
            return
        specs = self._agent.tool_registry.get_all_tools_config()
        if not specs:
            await self._note("no tools registered")
            return
        est = self.ledger.estimator
        costs = {name: est.tokens(len(json.dumps(spec, sort_keys=True))) for name, spec in specs.items()}
        lines = [f"{name}: ~{n} tokens" for name, n in sorted(costs.items(), key=lambda kv: -kv[1])]
        await self._note(f"tool schemas cost ~{sum(costs.values())} tokens on every call:\n" + "\n".join(lines))

    async def _decisions_command(self) -> None:
        entries = self.decisions.entries(limit=20)
        if not entries:
            await self._note("decision log is empty")
            return
        await self._note("decision log (most recent last):\n" + "\n".join(e.line() for e in entries))

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

    async def _switch_model(self, name: str, num_ctx: int | None = None, *, persist: bool = True) -> bool:
        """Unload the old model, then preflight and load the new one (at the profile's num_ctx unless given).

        False when the switch failed and the old model stays. ``persist=False`` is for a session swap, which
        saves the incoming session itself rather than stamping the outgoing one with the new model.
        """
        assert self._client is not None
        if name == self._model and (num_ctx is None or self._client.pinned(name) == num_ctx):
            await self._note(f"already on {name}")
            return True
        target = num_ctx or self.config.profile.num_ctx
        old = self._model
        await self._note(f"switching to {name} at num_ctx {target} (unloading {old or 'nothing'} first; this can take a while)")
        status = self.query_one(StatusBar)
        status.show_loading(name)
        if old is not None:
            await self._client.unload(old)
        try:
            result = await run_preflight(self._client, [Candidate(name, target)], env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            status.show_loading(None)
            await self._note(f"switch failed: {exc}")
            if old is not None:
                await self._note(f"staying on {old}; it reloads on the next prompt")
            return False
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
        self._preflight = result
        await self._build_helpers(result.model)
        if persist:
            self._save_session()
        await self._note(f"now on {result.model} at num_ctx {result.num_ctx} (budget {result.budget}); conversation kept")
        return True

    # --- sessions -------------------------------------------------------------------------------

    @property
    def _model_name(self) -> str:
        return self._model or self.config.profile.model

    @property
    def _num_ctx(self) -> int:
        pinned = self._client.pinned(self._model) if self._client is not None and self._model else None
        return pinned or self.config.profile.num_ctx

    def _calibration(self) -> dict[str, Any]:
        """What a resume needs to estimate like before: the learned token ratio, plus the preflight's budget."""
        out: dict[str, Any] = {"estimator": self.ledger.estimator.state()}
        if self._preflight is not None:
            out.update(budget=self._preflight.budget, weights=self._preflight.weights)
        return out

    def _save_session(self) -> None:
        """Snapshot the history and the session's model, num_ctx and calibration."""
        self.store.save_messages(self.session_id, self._messages)
        self.store.update(self.session_id, model=self._model_name, num_ctx=self._num_ctx, calibration=self._calibration())

    def _on_request(self, event: ModelRequestSent) -> None:
        pending = self.ledger.pending  # set by the ledger just before the main loop's call; None for summarizer calls
        num_ctx = int(event.body.get("options", {}).get("num_ctx", 0))
        self._open_calls[event.model] = self.store.record_call(
            self.session_id, event.model, num_ctx, event.body, pending.estimate if pending is not None else None)

    def _on_call_done(self, event: ModelCallCompleted) -> None:
        if self._watchdog is not None:
            self._watchdog.record_call(event.model, event.metrics, event.estimate)
        n = self._open_calls.pop(event.model, None)
        if n is not None:
            self.store.complete_call(self.session_id, n, event.metrics.prompt_eval_count, event.metrics.eval_count)

    async def _adopt(self, info: SessionInfo) -> None:
        """Make ``info`` the live session: its settled history, learned estimator, transcript and queue.

        Restored prompts come back paused, so nothing runs by itself after a crash or a switch.
        """
        messages, dropped = settle(self.store.messages(info.id))
        queued = self.store.queue(info.id)
        self.session_id = info.id
        self._messages[:] = messages
        if info.estimator_state:
            self.ledger.estimator.restore(info.estimator_state)
        self.ledger.records.clear()
        self._save_session()
        await self._render_history(messages)
        tail = f"; dropped {dropped} unfinished messages from a turn that never completed" if dropped else ""
        await self._note(f"session {info.id} ({info.title or 'untitled'}): {len(messages)} messages{tail}")
        if queued:
            self.queue.restore(queued)
            self._refresh_queue_label()
            await self._note(f"{len(queued)} queued prompts restored; Enter resumes, Esc clears")

    async def _render_history(self, messages: list[Message]) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.remove_children()
        for msg in messages:
            text = message_text(msg)
            if not text:
                continue
            if msg["role"] == "user":
                await chat.mount(Static(f"you: {text}", classes="user", markup=False))
            else:
                reply = Reply()
                await chat.mount(reply)
                await reply.finish(text)
        chat.scroll_end(animate=False)

    async def _switch_session(self, info: SessionInfo) -> None:
        """Leave the current session (saved) for ``info``, reloading the model if its model or num_ctx differ."""
        if self._client is not None and (info.model, info.num_ctx) != (self._model, self._num_ctx):
            if not await self._switch_model(info.model, info.num_ctx, persist=False):
                await self._note(f"could not load {info.model} at num_ctx {info.num_ctx}; staying in session {self.session_id}")
                return
        self._save_session()
        self.store.save_queue(self.session_id, ())  # prompts still queued move with the user to the new session
        await self._adopt(info)

    async def _sessions_command(self) -> None:
        rows = self.store.sessions(limit=20)
        if not rows:
            await self._note("no saved sessions yet")
            return
        await self._note("sessions (most recent first; * is this one):\n"
                         + "\n".join(r.line(current=r.id == self.session_id) for r in rows))

    async def _resume_command(self, args: str) -> None:
        if not args:
            await self._note("usage: /resume <id> (/sessions lists them)")
            return
        info = self.store.resolve(args)
        if info is None:
            await self._note(f"no single session matches '{args}' (/sessions lists them)")
        elif info.id == self.session_id:
            await self._note(f"already in session {info.id}")
        else:
            await self._switch_session(info)

    async def _fork_command(self, args: str) -> None:
        source = self.store.resolve(args) if args else self.store.get(self.session_id)
        if source is None:
            await self._note(f"no single session matches '{args}' (/sessions lists them)")
            return
        if source.id == self.session_id:
            self._save_session()
        fork = self.store.fork(source.id)
        await self._note(f"forked {source.id} as {fork.id}")
        await self._switch_session(fork)

    async def _inspect_command(self, args: str) -> None:
        if args and not args.isdigit():
            await self._note("usage: /inspect [n]")
            return
        self._open_inspector(int(args) if args else None)

    def _open_inspector(self, n: int | None = None) -> None:
        if not isinstance(self.screen, Inspector):
            self.push_screen(Inspector(self.store.calls(self.session_id), n))

    def action_inspect(self) -> None:
        self._open_inspector()

    def _pump(self) -> None:
        """Start the next queued prompt if idle, not paused, no approval is pending and no red Alert is active."""
        if self._busy or self.approval_pending or self.held_by_alert or (self._client is not None and self._agent is None):
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
        if self.tev is not None:
            self.run_worker(self._route(prompt), name="route", group="decide")
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
            if self._agent is None:  # test mode: the replier adds no messages itself
                self.fake_messages.extend([
                    {"role": "user", "content": [{"text": prompt}]},
                    {"role": "assistant", "content": [{"text": reply.text}]},
                ])
            try:
                await self._compact_after_turn()
            except Exception as exc:
                await self._note(f"compaction error: {exc}")
            self._save_session()
        finally:
            if run_id == self._run_id:  # a cancelled run may be unwound after the app has moved on
                self._busy = False
                self._pump()

    def action_cancel(self) -> None:
        """Cancel the generation and pause the queue so the next prompt never starts by itself.

        An open approval prompt is denied first, so the tool call it guards is cancelled, never run.
        """
        for prompt in self.query(ApprovalPrompt):
            prompt.decide(False)
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
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("config", nargs="?", type=Path, help="path to harness.toml (default ./harness.toml)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--resume", metavar="ID", help="resume the session with this id (or unambiguous prefix)")
    which.add_argument("--continue", dest="continue_last", action="store_true", help="resume the most recent session")
    args = parser.parse_args(argv)
    HarnessApp(load_config(args.config), resume=args.resume, continue_last=args.continue_last).run()
