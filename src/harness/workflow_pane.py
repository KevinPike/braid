"""Workflows in the TUI (M15): ``/plan``, ``/build`` and ``/runs`` for the chat app.

The pane owns one run at a time. It reuses the CLI's pieces (``drive_graph``, ``Progress``, ``describe``,
``final_report``, ``show``) so the TUI and ``harness-workflow`` cannot disagree, and puts everything a human sees
into the chat log: a progress line per node that updates in place, sandboxed commands with their exit code and
duration, and every question as a widget that collapses to a record once answered. A run holds the Prompt queue from
start to finish. The graph's state, including the Recorder, is checkpointed in SQLite, so a cancelled or killed run
resumes from its last checkpoint.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.graph import StateGraph
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from harness.approval import APPLY_WORKSPACE, ApprovalRequest
from harness.chat import AnswerPrompt
from harness.debuglog import DebugLog
from harness.guard.backend import Backend
from harness.guard.preflight import Candidate, PreflightError, run_preflight, system_gpu_limit
from harness.hooks import ApprovalRequested, ApprovalResolved, WorkflowInterrupted, WorkflowNodeFinished
from harness.inspector import RunInspector
from harness.sandbox import RunResult, Sandbox, SandboxPolicy, SeatbeltSandbox, Workspace
from harness.status import StatusBar
from harness.workflow.backend import choose, make_model, make_roles
from harness.workflow.cli import api_error_note, cost_note, plan_id_for, show
from harness.workflow.execute import build_execution_graph, drive_graph, initial_state
from harness.workflow.model import WorkflowModel
from harness.workflow.plan import Plan
from harness.workflow.plan_loop import build_plan_graph, initial_plan_state
from harness.workflow.progress import PLAN_NODES, WORK_NODES, Progress, describe, final_report
from harness.workflow.records import Recorder, RunLog, read_log
from harness.workflow.runs import RunInfo, RunStore, open_checkpointer

if TYPE_CHECKING:
    from harness.app import HarnessApp

COMMAND_WIDTH = 90


@dataclass
class WorkflowDeps:
    """What the pane needs from outside the app; tests replace the Ollama-facing parts and the clock."""

    model_factory: Callable[[str], WorkflowModel] | None = None  # default: the app's guard client
    sandbox: Sandbox | None = None  # default: Seatbelt
    backend: Backend | None = None  # default: the app's guard client; None in both means no model swap
    chat_model: str | None = None  # the chat model when the app has no client of its own
    chat_num_ctx: int | None = None
    clock: Callable[[], float] = time.monotonic
    gpu_limit: Callable[[], int] = system_gpu_limit
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    refresh_s: float = 0.25  # how often the live line and the status bar clock redraw


def shorten_command(command: str) -> str:
    """``pytest -q ...`` rather than the venv's full interpreter path."""
    text = re.sub(r"^\S*python\S* -m ", "", command.strip())
    text = text.replace("\n", " ")
    return text if len(text) <= COMMAND_WIDTH else text[: COMMAND_WIDTH - 1] + "…"


class ReportingSandbox:
    """A Sandbox that tells ``report`` about each finished command, so the chat can show it inline."""

    def __init__(self, inner: Sandbox, report: Callable[[str, RunResult], None]) -> None:
        self.inner, self.report = inner, report

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = 120.0, env: Mapping[str, str] | None = None,
    ) -> RunResult:
        result = await self.inner.run(command, cwd=cwd, policy=policy, timeout_s=timeout_s, env=env)
        self.report(command, result)
        return result


def parse_flags(
    args: str, values: tuple[str, ...] = ("--resume", "--model"), switches: tuple[str, ...] = ("--no-hints",),
) -> tuple[dict[str, str], str]:
    """Leading ``--flag [value]`` words, then the rest as one string."""
    words = args.split()
    flags: dict[str, str] = {}
    while words and words[0].startswith("--"):
        word = words.pop(0)
        if word in switches:
            flags[word] = "1"
        elif word in values and words:
            flags[word] = words.pop(0)
        else:
            raise ValueError(f"unknown or incomplete option {word}")
    return flags, " ".join(words)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


@dataclass
class _Live:
    node: str
    started: float
    detail: str
    widget: Static


@dataclass
class _Setup:
    graph: StateGraph[Any]
    state: Any
    finish: Callable[[dict[str, Any], float], tuple[str, str]]  # (final state, total seconds) -> (chat text, outcome)


class WorkflowPane:
    def __init__(self, app: HarnessApp, deps: WorkflowDeps | None = None) -> None:
        self.app = app
        self.deps = deps or WorkflowDeps()
        self._store: RunStore | None = None
        self.running = False
        self.run_id: str | None = None
        self.recorder = Recorder()
        self.last_report = ""
        self.last_total = 0.0
        self.debug: DebugLog | None = None
        self.progress = Progress(io.StringIO(), io.StringIO(), clock=self.deps.clock, tty=False)
        self._actions: asyncio.Queue[Callable[[], Awaitable[None]]] = asyncio.Queue()
        self._live: dict[str, list[_Live]] = defaultdict(list)
        self._records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._starts: dict[str, float] = {}
        self._elapsed_before = 0.0
        self._session_start = 0.0
        self._idx, self._total = 0, 0
        self._swapped = False
        self._chat: tuple[str | None, int] = (None, 0)
        self._active_model = ""

    # --- bookkeeping -----------------------------------------------------------------------------------------
    @property
    def store(self) -> RunStore:
        if self._store is None:
            self._store = RunStore(self.app.config.workflow.checkpoints_db.expanduser())
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def elapsed(self) -> float:
        """Active seconds of this run, including the sessions before a resume."""
        return self._elapsed_before + (self.deps.clock() - self._session_start)

    def _backend(self) -> Backend | None:
        return self.deps.backend or self.app._client

    def _runs_dir(self) -> Path:
        return self.app.config.workflow.runs_dir.expanduser()

    # --- UI actions, applied in order by one consumer ------------------------------------------------------------
    def _ui(self, action: Callable[[], Awaitable[None]]) -> None:
        self._actions.put_nowait(action)

    async def _consume(self) -> None:
        while True:
            action = await self._actions.get()
            try:
                await action()
            except Exception:  # the chat disappearing under a late action must not end the run
                pass
            finally:
                self._actions.task_done()

    async def _barrier(self) -> None:
        """Wait until every UI action queued so far has been applied."""
        await self._actions.join()

    async def _mount(self, widget: Widget) -> None:
        chat = self.app.query_one("#chat", VerticalScroll)
        await chat.mount(widget)
        chat.scroll_end(animate=False)

    def _line(self, text: str, classes: str = "note") -> None:
        self._ui(lambda: self._mount(Static(text, classes=classes, markup=False)))

    def _live_text(self, live: _Live) -> str:
        p = self.progress
        p.node, p.started, p.detail = live.node, live.started, live.detail
        return p.line()

    async def _refresh(self) -> None:
        for lives in self._live.values():
            for live in lives:
                live.widget.update(self._live_text(live))
        self.app.query_one(StatusBar).set_run_elapsed(self.elapsed())

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.deps.refresh_s)
            self.progress.frame += 1
            self._ui(self._refresh)

    def _drop_live(self) -> None:
        """Remove the running-node lines (the run is waiting for a human, or over)."""
        gone = [live.widget for lives in self._live.values() for live in lives]
        self._live.clear()

        async def remove() -> None:
            for widget in gone:
                await widget.remove()

        self._ui(remove)

    # --- graph callbacks ---------------------------------------------------------------------------------------
    def _on_start(self, node: str) -> None:
        p = self.progress
        p.begin(node)
        live = _Live(node, p.started, p.detail, Static("", classes="note live", markup=False))
        live.widget.update(self._live_text(live))
        self._live[node].append(live)
        self._starts[node] = p.started
        self._ui(lambda: self._mount(live.widget))

    def _on_record(self, record: dict[str, Any]) -> None:
        """A node wrote its log line: the live line becomes its permanent result line, in place."""
        node = record["node"]
        text = describe(record)
        self._records[node].append(record)
        lives = self._live.get(node)
        live = lives.pop(0) if lives else None

        async def finish() -> None:
            if live is not None:
                live.widget.update(text)
                live.widget.remove_class("live")
            else:
                await self._mount(Static(text, classes="note", markup=False))

        self._ui(finish)

    def _on_update(self, node: str, update: dict[str, Any]) -> None:
        records = self._records.get(node)
        record = records.pop(0) if records else None
        now = self.deps.clock()
        if record and record.get("node_seconds") is not None:
            duration = float(record["node_seconds"])
        else:
            duration = now - self._starts.get(node, now)
        attempt = int(record["attempt"]) if record and record.get("attempt") is not None else 1
        if "idx" in update:
            self._idx = update["idx"]
        if "plan" in update:
            self._total = len(update["plan"]["behaviors"])
        if ("idx" in update or "plan" in update) and self._total:
            self.progress.set_behavior(self._idx, self._total)
        if "round" in update:
            self.progress.set_round(update["round"], self.app.config.workflow.max_rounds)
        assert self.run_id is not None
        self.app.hooks.publish(WorkflowNodeFinished(self.run_id, node, duration, attempt))

    def _on_command(self, command: str, result: RunResult) -> None:
        code = "timed out" if result.timed_out else f"exit {result.exit_code}"
        self._line(f"$ {shorten_command(command)}  {code}  {result.duration_s:.1f}s", "note command")

    # --- questions to the human --------------------------------------------------------------------------------
    @staticmethod
    def _label(payload: dict[str, Any]) -> str:
        kind = str(payload["kind"])
        if kind == "stuck":
            return f"stuck on: {str(payload['behavior'])[:60]}"
        if kind == "review":
            return f"review of the Plan, round {payload['round']}/{payload['max_rounds']}"
        return {"clarify": "clarify the Plan", "sign_off": "sign-off"}.get(kind, kind)

    async def _ask(self, payload: dict[str, Any]) -> str:
        assert self.run_id is not None
        kind = payload["kind"]
        self.app.hooks.publish(WorkflowInterrupted(self.run_id, kind))
        self.store.update(self.run_id, state="waiting")
        self._drop_live()
        await self._barrier()
        try:
            if kind == "finish":
                return await self._ask_finish(payload)
            return await self._ask_text(payload)
        finally:
            self.store.update(self.run_id, state="running")

    async def _ask_finish(self, payload: dict[str, Any]) -> str:
        """The real Approval path: it emits ApprovalRequested and ApprovalResolved like a gated tool call."""
        request = ApprovalRequest(APPLY_WORKSPACE, {k: payload[k] for k in ("workspace", "changes", "diff")})
        hooks = self.app.hooks
        hooks.publish(ApprovalRequested(request))
        try:
            approved = await self.app.ask_approval(request)
        except asyncio.CancelledError:
            hooks.publish(ApprovalResolved(request, False))  # a cancelled run applies nothing and asks again on resume
            raise
        hooks.publish(ApprovalResolved(request, approved))
        return "yes" if approved else "no"

    async def _ask_text(self, payload: dict[str, Any]) -> str:
        app = self.app
        buffer = io.StringIO()
        show(payload, buffer)
        lines = buffer.getvalue().strip("\n").splitlines()
        hint = lines.pop().lstrip()[1:].strip() if lines and lines[-1].lstrip().startswith(">") else ""
        widget = AnswerPrompt("\n".join(lines), hint)
        label = self._label(payload)
        previous = app.focused
        app._pauses_waiting += 1
        app._refresh_queue_label()
        outcome = "cancelled"
        try:
            await self._mount(widget)
            widget.focus_field()
            answer = await widget.answer()
            outcome = answer[:80]
            return answer
        finally:
            app._pauses_waiting -= 1
            await app._collapse(widget, f"{'✗' if outcome == 'cancelled' else '✓'} {label}? {outcome}", previous)
            app._refresh_queue_label()

    # --- swapping the model around a run ------------------------------------------------------------------------
    def _chat_target(self) -> tuple[str | None, int]:
        app = self.app
        if app._client is not None:
            return app._model, app._num_ctx
        return self.deps.chat_model, self.deps.chat_num_ctx or app.config.profile.num_ctx

    async def _enter(self, model_id: str) -> bool:
        """Unload the chat model and preflight the workflow model, behind the status bar's loading state."""
        app = self.app
        backend = self._backend()
        if backend is None or choose(app.config, model=model_id).backend == "anthropic":
            return True  # an API run skips Preflight and the swap: the chat model stays loaded (ADR 0006)
        chat_model, chat_ctx = self._chat_target()
        if app._client is not None and chat_model is None:
            await app.note("workflow: the chat model is still loading; try again when preflight is done")
            return False
        wf_ctx = app.config.workflow.num_ctx
        if chat_model == model_id and chat_ctx == wf_ctx:
            return True
        self._chat, self._swapped = (chat_model, chat_ctx), True
        app.query_one(StatusBar).show_loading(model_id)
        await app.note(
            f"workflow: loading {model_id} at num_ctx {wf_ctx} (unloading {chat_model or 'nothing'} first; this can take a while)")
        try:
            if chat_model and chat_model != model_id:
                await backend.unload(chat_model)
            result = await run_preflight(backend, [Candidate(model_id, wf_ctx)], env=self.deps.env, gpu_limit=self.deps.gpu_limit())
        except PreflightError as exc:
            await app.note(f"workflow: {model_id} did not load: {exc}")
            return False
        for check in result.checks:
            if check.severity.value in ("warn", "fail"):
                await app.note(f"preflight {check.severity.value}: {check.detail}")
        if app._watchdog is not None:
            app._watchdog.set_expected_size(result.model, result.weights)
            app._watchdog.set_active(result.model)
        await app._end_loading()
        return True

    async def _leave(self) -> None:
        """Whatever way the run ended, the workflow model goes and the chat model is preflighted again."""
        app = self.app
        backend = self._backend()
        if not self._swapped or backend is None:
            return
        self._swapped = False
        chat_model, chat_ctx = self._chat
        try:
            app.query_one(StatusBar).show_loading(chat_model or "chat model")
            await app.note(f"workflow: restoring {chat_model or 'the chat model'}")
            if self._active_model != chat_model:
                await backend.unload(self._active_model)
            if chat_model:
                result = await run_preflight(backend, [Candidate(chat_model, chat_ctx)], env=self.deps.env, gpu_limit=self.deps.gpu_limit())
                if app._client is not None:
                    app._preflight = result
                if app._watchdog is not None:
                    app._watchdog.set_expected_size(result.model, result.weights)
                    app._watchdog.set_active(result.model)
        except PreflightError as exc:
            await app.note(f"workflow: {chat_model} did not reload: {exc}; it loads on the next prompt")
        except Exception as exc:
            await app.note(f"workflow: could not restore the chat model: {exc}")
        finally:
            try:
                await app._end_loading()
            except Exception:
                pass

    # --- running --------------------------------------------------------------------------------------------------
    def _model_for(self, model_id: str) -> WorkflowModel | None:
        if self.deps.model_factory is not None:
            return self.deps.model_factory(model_id)
        config = self.app.config
        return make_model(config, choose(config, model=model_id), self.app._client)

    def _resumed(self, values: dict[str, Any]) -> None:
        data = values.get("recorder")
        if data:
            self.recorder.load(data)
            self._elapsed_before = float(data["elapsed_s"])
        self._idx = values.get("idx", 0)
        if "plan" in values:
            self._total = len(values["plan"]["behaviors"])
            self.progress.set_behavior(self._idx, self._total)
        if values.get("round"):
            self.progress.set_round(values["round"], self.app.config.workflow.max_rounds)

    async def _run(
        self, info: RunInfo, *, resume: bool, setup: Callable[[WorkflowModel, Sandbox, RunLog, Recorder], Awaitable[_Setup | None]],
    ) -> None:
        app = self.app
        self.running = True
        self.run_id = info.run_id
        self._active_model = info.model
        self._elapsed_before, self._idx, self._total = 0.0, 0, 0
        self._live.clear()
        self._records.clear()
        self._starts.clear()
        self.progress = Progress(
            io.StringIO(), io.StringIO(), clock=self.deps.clock, tty=False, labelled=PLAN_NODES if info.kind == "plan" else WORK_NODES)
        if info.kind == "plan":
            self.progress.set_round(1, app.config.workflow.max_rounds)
        if self.debug is not None:
            self.debug.active = True
        consumer: asyncio.Future[None] | None = None
        ticker: asyncio.Future[None] | None = None
        try:
            if resume:
                self.store.update(info.run_id, state="running")
            else:
                self.store.create(info)
            if not await self._enter(info.model):
                self.store.update(info.run_id, state="failed", outcome="model did not load")
                return
            model = self._model_for(info.model)
            if model is None:
                await app.note("workflow: no model backend in this session")
                self.store.update(info.run_id, state="failed", outcome="no model backend")
                return
            self.recorder = rec = Recorder()
            log = RunLog(self._runs_dir() / f"{info.run_id}.jsonl", listener=self._on_record)
            sandbox = ReportingSandbox(self.deps.sandbox or SeatbeltSandbox(), self._on_command)
            ready = await setup(model, sandbox, log, rec)
            if ready is None:
                self.store.update(info.run_id, state="failed", outcome="setup failed")
                return
            self._session_start = self.deps.clock()
            consumer = asyncio.ensure_future(self._consume())
            ticker = asyncio.ensure_future(self._tick())
            app.query_one(StatusBar).set_run_elapsed(0.0)
            await app.note(f"{info.kind} {info.run_id} started with {info.model}; Ctrl-C pauses it at the last checkpoint")
            async with open_checkpointer(app.config.workflow.checkpoints_db.expanduser()) as saver:
                final = await drive_graph(
                    ready.graph, ready.state, self._ask, on_update=self._on_update, on_start=self._on_start,
                    thread_id=info.run_id, checkpointer=saver, resume=resume, on_resume=self._resumed,
                )
            self._drop_live()
            total = self.elapsed()
            if final.get("status") == "api_error":  # F8: resumable from the Checkpoint before the failed node
                self.store.update(info.run_id, state="failed", outcome=f"api_error: {final['api_error']['error']}")
                self._line(f"{info.kind} {info.run_id} {api_error_note(final['api_error'])}.\n"
                           f"Resume from its last checkpoint: /{info.kind} --resume {info.run_id}")
                return
            text, outcome = ready.finish(final, total)
            self.last_total = total
            self.last_report = text
            self._line(text, "note table")
            self.store.update(info.run_id, state="finished", outcome=outcome)
        except asyncio.CancelledError:
            self.store.update(info.run_id, state="paused")
            self._line(f"{info.kind} {info.run_id} paused at its last checkpoint; nothing was applied. Resume: /{info.kind} --resume {info.run_id}")
            raise
        except Exception as exc:
            self.store.update(info.run_id, state="failed", outcome=f"{type(exc).__name__}: {exc}"[:200])
            self._line(f"{info.kind} {info.run_id} failed: {exc}\nResume from its last checkpoint: /{info.kind} --resume {info.run_id}")
        finally:
            if ticker is not None:
                ticker.cancel()
            self._drop_live()
            if consumer is not None:
                try:
                    await asyncio.wait_for(self._barrier(), 5)  # not forever: the chat may be shutting down underneath us
                except TimeoutError:
                    pass
                consumer.cancel()
            try:
                app.query_one(StatusBar).set_run_elapsed(None)
            except Exception:
                pass
            try:
                await self._leave()
            finally:
                if self.debug is not None:
                    self.debug.active = False
                self.running = False

    # --- commands -----------------------------------------------------------------------------------------------
    async def command(self, name: str, args: str) -> None:
        try:
            if name == "runs":
                await self.runs(args)
            elif name == "plan":
                await self.plan(args)
            elif name == "build":
                await self.build(args)
        except ValueError as exc:
            await self.app.note(f"/{name}: {exc}")

    def _pick_model(self, flags: Mapping[str, str]) -> str:
        return choose(self.app.config, model=flags.get("--model")).model_id

    @staticmethod
    def _new_id(model_id: str, prefix: str = "") -> str:
        return f"{prefix}{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug(model_id)}"

    def _project(self) -> Path:
        return Path(self.app.config.tools.root).expanduser().resolve()

    def _resumable(self, flags: Mapping[str, str], kind: str) -> RunInfo | None:
        run = flags.get("--resume")
        if run is None:
            return None
        info = self.store.resolve(run)
        if info is None:
            raise ValueError(f"no single run matches '{run}' (/runs lists them)")
        if info.kind != kind:
            raise ValueError(f"{info.run_id} is a {info.kind} run; resume it with /{info.kind} --resume {info.run_id}")
        if info.state == "finished":
            raise ValueError(f"{info.run_id} already finished ({info.outcome})")
        return info

    def _plan_file(self, ref: str) -> Path:
        plans = self.app.config.workflow.plans_dir.expanduser()
        if not ref:
            found = sorted(plans.glob("*.json"), key=lambda p: p.stat().st_mtime) if plans.is_dir() else []
            if not found:
                raise ValueError(f"no Plans in {plans}; make one with /plan <goal>")
            return found[-1]
        for candidate in (Path(ref).expanduser(), plans / ref, plans / f"{ref}.json"):
            if candidate.is_file():
                return candidate
        matches = sorted(plans.glob(f"{ref}*.json")) if plans.is_dir() else []
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"no single Plan matches '{ref}' in {plans}")

    async def build(self, args: str) -> None:
        flags, rest = parse_flags(args)
        project = self._project()
        resumed = self._resumable(flags, "build")
        plan: Plan | None = None
        if resumed is not None:
            info = resumed
        else:
            path = self._plan_file(rest)
            plan = Plan.load(path)
            model_id = self._pick_model(flags)
            info = RunInfo(
                self._new_id(model_id), "build", plan.goal, str(project), model_id, plan_path=str(path), hints="--no-hints" not in flags,
            )
        wf = self.app.config.workflow

        async def setup(model: WorkflowModel, sandbox: Sandbox, log: RunLog, rec: Recorder) -> _Setup | None:
            app = self.app
            if plan is not None:
                workspace = await Workspace.create(project, wf.workspaces_dir.expanduser(), info.run_id)
                self.store.update(info.run_id, workspace=str(workspace.path))
                await app.note(f"plan: {plan.goal}\nworkspace: {workspace.path}\nrun log: {log.path}")
                if workspace.warning:
                    await app.note(f"warning: {workspace.warning}")
                synced = await workspace.sync_deps()
                if synced.exit_code != 0:
                    await app.note(f"uv sync --frozen failed:\n{synced.output}")
                    return None
                self._total = len(plan.behaviors)
                self.progress.set_behavior(0, self._total)
            else:
                workspace = await Workspace.attach(Path(info.project), Path(info.workspace))
                await app.note(f"resuming {info.run_id} in {workspace.path}; run log: {log.path}")
            graph = build_execution_graph(
                model, sandbox, workspace, wf, hints=info.hints, recorder=rec, log=log, **make_roles(app.config, info.model),
            )

            def finish(final: dict[str, Any], total: float) -> tuple[str, str]:
                status = str(final.get("status"))
                return final_report(rec, total, by_role=wf.roles.separated) + f"\n\nstatus: {status}; polish {rec.polish}", status

            return _Setup(graph, initial_state(plan) if plan is not None else None, finish)

        await self._run(info, resume=resumed is not None, setup=setup)

    async def plan(self, args: str) -> None:
        flags, goal = parse_flags(args)
        resumed = self._resumable(flags, "plan")
        project = self._project()
        wf = self.app.config.workflow
        if resumed is None:
            if not goal:
                raise ValueError("usage: /plan <goal> | /plan --resume <run-id>")
            model_id = self._pick_model(flags)
            plan_id = plan_id_for(goal, datetime.now().strftime("%Y%m%d-%H%M%S"))
            path = wf.plans_dir.expanduser() / f"{plan_id}.json"
            info = RunInfo(self._new_id(model_id, "plan-"), "plan", goal, str(project), model_id, plan_path=str(path))
        else:
            info = resumed

        async def setup(model: WorkflowModel, sandbox: Sandbox, log: RunLog, rec: Recorder) -> _Setup | None:
            await self.app.note(f"goal: {info.goal}\nproject: {info.project}\nrun log: {log.path}")
            graph = build_plan_graph(model, Path(info.project), wf, plan_id=Path(info.plan_path).stem, recorder=rec, log=log)

            def finish(final: dict[str, Any], total: float) -> tuple[str, str]:
                if final.get("status") == "spend_cap":
                    return f"stopped at the Spend cap (${rec.cost_usd:.4f} > ${wf.max_cost_usd:.2f}); no Plan saved", "spend_cap"
                return (
                    f"{final['markdown']}\ntotal wall time {total:.1f}s, model time {rec.model_seconds:.1f}s over {rec.model_calls} "
                    f"calls{cost_note(rec)}, {final['round']} round(s), status: {final['status']}\nplan saved: {final['plan_path']}\n"
                    f"build it: /build {Path(final['plan_path']).stem}",
                    str(final["status"]),
                )

            return _Setup(graph, initial_plan_state(info.goal) if resumed is None else None, finish)

        await self._run(info, resume=resumed is not None, setup=setup)

    async def runs(self, args: str) -> None:
        app = self.app
        if not args:
            rows = self.store.list(20)
            if not rows:
                await app.note("no workflow runs yet (/plan <goal>, /build [plan-id])")
                return
            await app.note("runs (most recent first; /runs <id> opens the log):\n"
                           + "\n".join(r.line(live=self.running and r.run_id == self.run_id) for r in rows))
            return
        info = self.store.resolve(args)
        if info is None:
            await app.note(f"no single run matches '{args}' (/runs lists them)")
            return
        path = self._runs_dir() / f"{info.run_id}.jsonl"
        if not path.exists():
            await app.note(f"run {info.run_id} has no log at {path}")
            return
        if not isinstance(app.screen, RunInspector):
            app.push_screen(RunInspector(info.run_id, read_log(path)))
