"""The runs `harness-mcp` starts (F18, ADR 0013): background tasks over the usual Workflow machinery.

A run here is what the TUI's workflow pane makes (``build_plan_graph`` or ``build_execution_graph`` driven by
``drive_graph`` over the SQLite checkpointer, recorded in the run index) with three differences. Its Spend cap is the
figure ``SpendGuard`` granted. A human pause (Stuck, a Plan review, a clarify) leaves the run ``waiting`` until the client
answers it, and each such answer is a Route record with basis ``mcp-client``. Sign-off and the finish (applying the diff)
are never answered here: the run parks at that Checkpoint, and a human resumes it in the TUI. Nothing in this module
applies a diff.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO, TypeVar, cast

from harness.config import HarnessConfig
from harness.mcp.hub import redact
from harness.mcp.spend import ORIGIN, Refusal, SpendGuard
from harness.sandbox import Sandbox, Workspace
from harness.workflow import bench
from harness.workflow.backend import Choice, RoleModels
from harness.workflow.cli import Cell, api_error_note, plan_id_for, run_grid
from harness.workflow.execute import build_execution_graph, drive_graph, initial_state
from harness.workflow.model import WorkflowModel
from harness.workflow.plan import Plan, PlanAuthor
from harness.workflow.plan_loop import build_plan_graph, initial_plan_state
from harness.workflow.progress import describe, final_report
from harness.workflow.records import Recorder, RunLog
from harness.workflow.runs import RunInfo, RunStore, open_checkpointer

ANSWERABLE = ("review", "clarify", "stuck")  # the interrupts a client may answer (D21)
HUMAN_ONLY = ("sign_off", "finish")  # Sign-off, and the diff that would be applied: never over MCP
KEEP_LINES = 40  # progress lines kept per run
T = TypeVar("T")


class RunnerError(Exception):
    """A request the runner refuses; its text goes to the client."""


@dataclass(frozen=True)
class Models:
    """The models a run uses, vetted before the run is asked for: ``make`` gives a fresh run model, ``roles`` the Role models."""

    make: Callable[[], WorkflowModel]
    roles: RoleModels


@dataclass(frozen=True)
class RunRequest:
    """Everything a run needs, decided before it starts. ``config`` is the run's own copy, and ``config.workflow.max_cost_usd``
    is the Spend cap it was granted."""

    kind: Literal["plan", "build", "bench"]
    choice: Choice
    make: Callable[[], WorkflowModel]  # a fresh model per call (a bench makes one per run)
    roles: RoleModels
    config: HarnessConfig
    cap_usd: float
    goal: str = ""
    plan_path: Path | None = None  # build: the Plan to build
    hints: bool = True  # build: false is the baseline, where code answers Stuck and Sign-off
    bench_set: bench.BenchSet | None = None
    repeat: int = 1
    label: str = ""


class Runner(Protocol):
    async def start(self, request: RunRequest) -> str:
        """Record the run and start it in the background; returns its id. Does not wait for the run."""
        ...

    async def answer(self, run_id: str, answer: str) -> dict[str, Any]: ...

    def cancel(self, run_id: str) -> bool: ...


class Parked(Exception):
    """The run reached an interrupt only a human may answer; it stops at its Checkpoint."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


@dataclass
class Live:
    """A run this process owns: what the status tool shows beyond the run index."""

    info: RunInfo
    request: RunRequest
    rec: Recorder
    lines: list[str] = field(default_factory=list)
    waiting: dict[str, Any] | None = None  # {"kind", "payload"} while stopped for a human
    reply: asyncio.Future[str] | None = None  # set while the run waits for a client's answer
    report: str = ""
    task: asyncio.Task[None] | None = None
    saving: set[asyncio.Future[None]] = field(default_factory=set)  # cost updates still on their way to the run index
    tick: asyncio.Event = field(default_factory=asyncio.Event)

    def say(self, line: str) -> None:
        self.lines.append(redact(line))
        del self.lines[:-KEEP_LINES]
        self.poke()

    def poke(self) -> None:
        """Wake whoever waits on this run (``braid_run_status`` with ``wait_s``)."""
        old, self.tick = self.tick, asyncio.Event()
        old.set()

    @property
    def done(self) -> bool:
        return self.task is None or self.task.done()


class RunBoard:
    """The runs this process owns, by id."""

    def __init__(self) -> None:
        self.runs: dict[str, Live] = {}

    def get(self, run_id: str) -> Live | None:
        return self.runs.get(run_id)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


class _Lines(io.TextIOBase):
    """A bench's ``out``: each line it writes becomes a progress line, and stays off stdout, which is the protocol's."""

    def __init__(self, line: Callable[[str], None]) -> None:
        self.line, self.rest = line, ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        *whole, self.rest = (self.rest + text).split("\n")
        for line in whole:
            if line.strip():
                self.line(line)
        return len(text)


class Driver:
    """The real ``Runner``."""

    def __init__(
        self, store: RunStore, board: RunBoard, guard: SpendGuard, sandbox: Sandbox, *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store, self.board, self.guard, self.sandbox, self.clock = store, board, guard, sandbox, clock

    # --- the run index -------------------------------------------------------------------------------------
    # The index shares workflows.db with the checkpointer, whose writes finish on this event loop. A write that waits on
    # SQLite's lock with the loop blocked therefore waits for the very writer it needs: the store has a short busy
    # timeout, and a locked write is retried between awaits.
    async def _retry(self, op: Callable[[], T]) -> T:
        for _ in range(400):
            try:
                return op()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) and "busy" not in str(exc):
                    raise
                await asyncio.sleep(0.05)
        return op()

    async def _save(self, run_id: str, **changes: Any) -> None:
        await self._retry(lambda: self.store.update(run_id, **changes))

    async def _final(self, live: Live, **changes: Any) -> None:
        """The run's last write: its state and its final cost in one update, after the progress saves have landed, so a reader
        who sees the state also sees the cost (a locked index retries this write, never a separate cost write after it)."""
        await asyncio.gather(*live.saving, return_exceptions=True)
        await self._save(live.info.run_id, cost_usd=self._cost(live), **changes)

    def _save_soon(self, live: Live, **changes: Any) -> None:
        """A progress update from a synchronous listener: sent when the index is free, and always before the run's last write."""
        async def go() -> None:
            try:
                await self._save(live.info.run_id, **changes)
            except sqlite3.OperationalError:
                pass  # the closing write carries the figure

        task = asyncio.ensure_future(go())
        live.saving.add(task)
        task.add_done_callback(live.saving.discard)

    # --- starting ------------------------------------------------------------------------------------------
    async def start(self, request: RunRequest) -> str:
        wf = request.config.workflow
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id = request.choice.model_id
        prefix = {"plan": "plan-", "build": "", "bench": "bench-"}[request.kind]
        run_id = f"{prefix}{stamp}-{slug(model_id)}"
        while run_id in self.board.runs or self.store.get(run_id) is not None:  # two starts in one second
            run_id += "x"
        goal, plan_path = request.goal, ""
        if request.kind == "plan":
            plan_path = str(wf.plans_dir.expanduser() / f"{plan_id_for(request.goal, stamp)}.json")
        elif request.kind == "build":
            assert request.plan_path is not None
            plan_path = str(request.plan_path)
            goal = Plan.load(request.plan_path).goal
        else:
            assert request.bench_set is not None
            goal = f"bench {request.bench_set.name} x{request.repeat}, label {request.label}"
        # recorded before the run starts: its Spend cap is reserved against the daily ceiling from here on
        info = RunInfo(
            run_id, request.kind, goal, str(self._project(request)), model_id, plan_path=plan_path, hints=request.hints,
            cap_usd=request.cap_usd, origin=ORIGIN,
        )
        await self._retry(lambda: self.store.create(info))
        rec = Recorder()
        rec.human = "mcp-client"
        live = Live(info, request, rec)
        self.board.runs[run_id] = live
        live.task = asyncio.ensure_future(self._run(live))
        return run_id

    @staticmethod
    def _project(request: RunRequest) -> Path:
        return Path(request.config.tools.root).expanduser().resolve()

    # --- the run -------------------------------------------------------------------------------------------
    async def _run(self, live: Live) -> None:
        rq = live.request
        started = self.clock()
        try:
            if rq.kind == "bench":
                outcome, live.report = await self._bench(live)
            else:
                async with open_checkpointer(rq.config.workflow.checkpoints_db.expanduser()) as saver:
                    final = await (self._plan if rq.kind == "plan" else self._build)(live, saver)
                if final.get("status") == "api_error":  # F8: resumable from the Checkpoint before the failed node
                    await self._final(live, state="failed", outcome=redact(f"api_error: {final['api_error']['error']}")[:200])
                    live.report = redact(api_error_note(final["api_error"]))
                    live.say(live.report)
                    return
                outcome, live.report = self._finish(live, final, self.clock() - started)
            await self._final(live, state="finished", outcome=outcome)
            live.say(f"finished: {outcome}")
        except Parked as parked:
            await self._final(live, state="waiting", outcome=f"waiting: {parked.kind}")
            live.say(f"stopped for a human at {parked.kind}; resume it in the TUI")
        except asyncio.CancelledError:
            await self._final(live, state="paused", outcome="cancelled")
            live.waiting = None
            live.say("cancelled; resumable from its last Checkpoint in the TUI")
        except Exception as exc:  # a failed run is a state, never a crashed server
            await self._final(live, state="failed", outcome=redact(f"{type(exc).__name__}: {exc}")[:200])
            live.say(f"failed: {type(exc).__name__}: {exc}")
        finally:
            live.poke()

    def _cost(self, live: Live) -> float:
        if live.request.kind != "bench":
            return live.rec.cost_usd
        return bench_cost(bench.read_ledger(live.request.config.paths.bench_ledger), live.request.label)

    def _listener(self, live: Live) -> Callable[[dict[str, Any]], None]:
        def on_record(record: dict[str, Any]) -> None:
            self._save_soon(live, cost_usd=live.rec.cost_usd)
            live.say(describe(record).strip())

        return on_record

    async def _ask(self, live: Live, payload: dict[str, Any]) -> str:
        """A human pause. Stuck, review and clarify wait here for the client's answer; Sign-off and the finish end the task."""
        kind = str(payload["kind"])
        live.waiting = {"kind": kind, "payload": payload}
        await self._save(live.info.run_id, state="waiting", outcome=f"waiting: {kind}", cost_usd=live.rec.cost_usd)
        live.say(f"waiting for {'the client' if kind in ANSWERABLE else 'a human in the TUI'}: {kind}")
        if kind not in ANSWERABLE:
            raise Parked(kind)
        live.reply = asyncio.get_running_loop().create_future()
        return await live.reply

    async def _plan(self, live: Live, saver: Any) -> dict[str, Any]:
        rq, info = live.request, live.info
        wf = rq.config.workflow
        log = RunLog(wf.runs_dir.expanduser() / f"{info.run_id}.jsonl", listener=self._listener(live))
        graph = build_plan_graph(rq.make(), Path(info.project), wf, plan_id=Path(info.plan_path).stem, recorder=live.rec, log=log)
        final = cast(dict[str, Any], await drive_graph(
            graph, initial_plan_state(info.goal), lambda payload: self._ask(live, payload), thread_id=info.run_id, checkpointer=saver,
        ))
        if final.get("status") not in ("api_error", "spend_cap"):
            path = Path(final["plan_path"])
            authored = Plan.load(path)
            authored.author = PlanAuthor(
                model=rq.choice.model_id, backend=rq.choice.backend, effort=wf.effort,
                written_at=datetime.now().isoformat(timespec="seconds"),
            )
            authored.save(path)
        return final

    async def _build(self, live: Live, saver: Any) -> dict[str, Any]:
        rq, info = live.request, live.info
        wf = rq.config.workflow
        assert rq.plan_path is not None
        workspace = await Workspace.create(Path(info.project), wf.workspaces_dir.expanduser(), info.run_id)
        await self._save(info.run_id, workspace=str(workspace.path))
        live.say(f"Workspace {workspace.path}")
        synced = await workspace.sync_deps()
        if synced.exit_code != 0:
            raise RuntimeError(f"uv sync --frozen failed: {synced.output[-300:]}")
        log = RunLog(wf.runs_dir.expanduser() / f"{info.run_id}.jsonl", listener=self._listener(live))
        graph = build_execution_graph(
            rq.make(), self.sandbox, workspace, wf, hints=rq.hints, recorder=live.rec, log=log, **rq.roles,
        )
        return cast(dict[str, Any], await drive_graph(
            graph, initial_state(Plan.load(rq.plan_path)), lambda payload: self._ask(live, payload),
            thread_id=info.run_id, checkpointer=saver,
        ))

    async def _bench(self, live: Live) -> tuple[str, str]:
        rq = live.request
        assert rq.bench_set is not None
        ledger = rq.config.paths.bench_ledger

        def on_line(line: str) -> None:
            self._save_soon(live, cost_usd=bench_cost(bench.read_ledger(ledger), rq.label))
            live.say(line)

        out = _Lines(on_line)
        code = await run_grid(
            rq.bench_set, [Cell(rq.label, rq.choice, rq.config, rq.make)], sandbox=self.sandbox, repeat=rq.repeat,
            max_cost=rq.cap_usd, ledger=ledger, out=cast(TextIO, out), err=cast(TextIO, out),
        )
        outcome = {0: "done", 3: "spend_cap"}.get(code, f"exit {code}")
        lines = bench.read_ledger(ledger)
        return outcome, redact(bench.label_table(lines, rq.label)) if any(x["label"] == rq.label for x in lines) else outcome

    def _finish(self, live: Live, final: dict[str, Any], seconds: float) -> tuple[str, str]:
        rq, rec = live.request, live.rec
        status = str(final.get("status"))
        if rq.kind == "plan":
            if status == "spend_cap":
                return "spend_cap", f"stopped at the Spend cap (${rec.cost_usd:.4f} > ${rq.cap_usd:.2f}); no Plan saved"
            return status, redact(f"{final['markdown']}\nplan saved: {final['plan_path']}\nbuild it: braid_build_start(plan_id={Path(final['plan_path']).stem!r})")
        return status, redact(final_report(rec, seconds, by_role=rq.config.workflow.roles.separated) + f"\n\nstatus: {status}; polish {rec.polish}")

    # --- answering and cancelling --------------------------------------------------------------------------
    async def answer(self, run_id: str, answer: str) -> dict[str, Any]:
        live = self.board.get(run_id)
        if live is None:
            info = self.store.resolve(run_id)
            if info is not None and info.state == "waiting":
                raise RunnerError(f"{info.run_id} was started by another process; resume it in the TUI (/{info.kind} --resume {info.run_id})")
            raise RunnerError(f"{run_id} is not a run this server owns")
        kind = live.waiting["kind"] if live.waiting else None
        if kind in HUMAN_ONLY:
            what = "Sign-off" if kind == "sign_off" else "applying the diff"
            raise RunnerError(f"{what} is for a human and is never possible over MCP: resume {live.info.run_id} in the TUI (/build --resume {live.info.run_id})")
        if live.reply is None or live.reply.done() or kind not in ANSWERABLE:
            raise RunnerError(f"{live.info.run_id} is not waiting for an answer (use braid_run_status)")
        # the answer lets the run spend again: re-clamp its cap to what the daily ceiling still allows, counting its own cost
        wf = live.request.config.workflow
        granted = self.guard.clamp(wf.max_cost_usd, excluding=live.info.run_id)
        if isinstance(granted, Refusal) or granted.cap_usd <= live.rec.cost_usd + 1e-9:
            reason = granted.reason if isinstance(granted, Refusal) else "the daily ceiling leaves this run nothing more to spend"
            raise RunnerError(f"not resumed: {reason}")
        wf.max_cost_usd = granted.cap_usd
        await self._save(live.info.run_id, state="running", outcome="", cap_usd=granted.cap_usd)
        live.waiting = None
        live.say(f"client answered {kind}: {answer[:60]}")
        live.reply.set_result(answer)
        return {"run_id": live.info.run_id, "answered": kind, "cap_usd": granted.cap_usd}

    def cancel(self, run_id: str) -> bool:
        live = self.board.get(run_id)
        if live is None or live.done:
            return False
        assert live.task is not None
        live.task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop every run this server still owns, each at its last Checkpoint."""
        tasks = [live.task for live in self.board.runs.values() if live.task is not None and not live.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def bench_cost(lines: list[dict[str, Any]], label: str) -> float:
    """What the ledger's runs under ``label`` cost."""
    return sum(float(x["summary"].get("totals", {}).get("cost_usd", 0.0)) for x in lines if x.get("label") == label)


def json_safe(value: Any, limit: int = 4000) -> Any:
    """``value`` as JSON-friendly data with every string cut to ``limit`` characters (a Plan draft or a pytest tail is long)."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {str(k): json_safe(v, limit) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v, limit) for v in value]
    return value if value is None or isinstance(value, bool | int | float) else json.dumps(value, default=str)[:limit]
