"""``harness-workflow``: ``plan`` runs the Plan loop on a goal (M14), ``build`` runs the Execution loop on a Plan (M13),
``bench`` runs a manifest set of (kata, Plan) pairs N times into the ledger and ``report`` reads it back (F2); with
``--models``/``--effort`` it runs a grid, a label per cell, and ``--batch`` sends the calls as Message Batches (F12)."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast, get_args

from harness.anthropic_api import make_client
from harness.config import Effort, HarnessConfig, arbiter_role, load_config
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, run_preflight, system_gpu_limit
from harness.sandbox import Sandbox, SeatbeltSandbox, Workspace
from harness.workflow import bench
from harness.workflow.backend import Choice, choose, make_arbiter, make_model
from harness.workflow.batch import BatchedWorkflowModel, Batcher
from harness.workflow.execute import Ask, execution_variant, build_execution_graph, drive_graph, initial_state
from harness.workflow.model import WorkflowModel, WorkflowModelError
from harness.workflow.plan import Plan
from harness.workflow.plan_loop import build_plan_graph, initial_plan_state
from harness.workflow.progress import PLAN_NODES, Progress, describe, final_report
from harness.workflow.records import Recorder, RunLog, amend_summary, write_summary
from harness.usage import require_pricing


CANCELLED_EXIT = 130  # 128 + SIGINT
SPEND_CAP_EXIT = 3  # the run passed max_cost_usd and stopped; nothing was applied or saved
API_ERROR_EXIT = 4  # an API request failed past the SDK's retries (F8); nothing was applied, the Workspace is kept


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness-workflow", description="Plan a goal, then build the Plan.")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("plan", help="turn a goal into an approved Plan through research, Critiques and your review")
    g.add_argument("goal", help="what to build, in a sentence or two")
    g.add_argument("--project", type=Path, default=Path("."), help="the project to research (default .)")
    g.add_argument("--model", help="model id (default: the backend's model from config; a claude- id picks the anthropic backend)")
    g.add_argument("--backend", choices=["ollama", "anthropic"], help="model backend (default [workflow] backend, or by the model id)")
    g.add_argument("--max-cost", type=float, metavar="USD", help="Spend cap for an API run (default [workflow] max_cost_usd)")
    g.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    g.add_argument("--unload", action="store_true", help="unload the model at the end")
    b = sub.add_parser("build", help="make each Behavior of a Plan Red then Green in a sandboxed Workspace")
    b.add_argument("--plan", type=Path, required=True, help="Plan JSON")
    b.add_argument("--project", type=Path, default=Path("."), help="the git project to build in (default .)")
    b.add_argument("--model", help="model id (default: the backend's model from config; a claude- id picks the anthropic backend)")
    b.add_argument("--backend", choices=["ollama", "anthropic"], help="model backend (default [workflow] backend, or by the model id)")
    b.add_argument("--max-cost", type=float, metavar="USD", help="Spend cap for an API run (default [workflow] max_cost_usd)")
    b.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    b.add_argument("--no-hints", action="store_true", help="baseline: Stuck skips the Behavior, clarify waives, Sign-off says yes")
    b.add_argument("--unload", action="store_true", help="unload the model at the end")
    n = sub.add_parser("bench", help="run a manifest set of (kata, Plan) pairs N times, --no-hints, into the ledger")
    n.add_argument("set", help="a set in the manifest (bench/katas.toml): claude-plans, local-plans, exotic, ...")
    n.add_argument("--model", help="model id (default: the backend's model from config)")
    n.add_argument("--backend", choices=["ollama", "anthropic"], help="model backend (default [workflow] backend, or by the model id)")
    n.add_argument("--repeat", type=int, default=1, help="runs per pair (default 1)")
    n.add_argument("--label", help="the ledger label (default: <set>-<model>-<timestamp>)")
    n.add_argument("--max-cost", type=float, default=1.0, metavar="USD",
                   help="stop the whole bench once it has cost this much (default 1.00); each run keeps its own Spend cap")
    n.add_argument("--manifest", type=Path, default=bench.DEFAULT_MANIFEST, help="the manifest (default bench/katas.toml)")
    n.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    n.add_argument("--no-red-check", action="store_true", help="F3 ablation: no Invariant check after Red, only on a Dispute")
    n.add_argument("--hide-rules", action="store_true", help="F3 ablation: do not show the Plan's Invariants to the models")
    n.add_argument("--models", metavar="A,B,...", help="F12 sweep: these model ids, one label per model x effort cell")
    n.add_argument("--effort", metavar="E,F,...", help="F12 sweep: these efforts (low, medium, high, xhigh, max)")
    n.add_argument("--batch", action="store_true",
                   help="F12: run every run of the grid at once and send their calls as Message Batches (half price)")
    n.add_argument("--parallel", type=int, metavar="N", help="runs at a time (default 1; with --batch, every run)")
    n.add_argument("--flush-s", type=float, default=30.0, metavar="S",
                   help="with --batch: send a batch S seconds after its first call even if some runs are not waiting (default 30)")
    r = sub.add_parser("report", help="the benchmark ledger: one label, a comparison of two, or the latest summary per kata")
    r.add_argument("labels", nargs="*", help="a label to show (none: list the labels)")
    r.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare label B against label A")
    r.add_argument("--frontier", nargs="+", metavar="LABEL", help="F12: hidden pass rate against cost per kata, per label")
    r.add_argument("--summaries", type=Path, metavar="PRACTICE_DIR", help="the latest summary per model x kata under DIR/*/runs/")
    r.add_argument("--all", action="store_true", help="with --summaries: include hinted runs")
    r.add_argument("--ledger", type=Path, help="the ledger (default <data_dir>/bench.jsonl)")
    r.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    return p


def show(payload: dict[str, Any], out: TextIO) -> None:
    def w(text: str = "") -> None:
        out.write(text + "\n")

    w(f"\n  HUMAN: {payload['kind']}")
    if payload["kind"] == "review":
        plan = Plan.model_validate({"goal": payload["goal"], **payload["draft"], "open_questions": payload["open_questions"]})
        w(f"  | round {payload['round']}/{payload['max_rounds']}")
        for line in plan.to_markdown().splitlines():
            w(f"  | {line}")
        for c in payload["critiques"]:
            for point in c["points"]:
                w(f"  | [{c['angle']}] {point}")
        for f in payload["unverified"]:
            w(f"  | unverified finding: {f['claim']} ({f['evidence']}: {f['problem']})")
        w("  > approve | research: <what to look into> | <anything else = notes for a revised draft>")
    elif payload["kind"] == "clarify":
        for f in payload["failed"]:
            w(f"  | assumption failed: {f}")
        for q in payload["open_questions"]:
            w(f"  | open question:     {q}")
        w("  > <how to proceed> | abort")
    elif payload["kind"] == "stuck":
        w(f"  | stuck on: {payload['behavior']}")
        for line in payload["last_run"].splitlines()[-8:]:
            w(f"  |   {line}")
        if dispute := payload.get("dispute"):  # F3: the implementer said a test is wrong; what the Invariant check found
            w(f"  | disputed {dispute['test']}: {dispute['reason']}")
            if dispute["upheld"]:
                w("  |   upheld by: " + "; ".join(v["invariant"] for v in dispute["violations"]))
            elif dispute.get("checked"):
                w(f"  |   not upheld: {dispute['checked']} checks against the Plan's Invariants passed")
            else:
                w("  |   not upheld: no Invariant covered it (add one to the Plan, or fix the test, then resume)")
            for u in dispute.get("unchecked", [])[:3]:
                w(f"  |   unchecked: {u['invariant']}: {u['error']}")
            if ruling := dispute.get("ruling"):  # F10: what the Arbiter ruled
                w(f"  |   the Arbiter ({ruling['model']}) ruled {ruling['verdict']}: {ruling['reason']}")
        show_proposed(payload, w)
        w("  > <a hint for the model> | skip | abort")
    elif payload["kind"] == "sign_off":
        w("  | demo output:")
        for line in payload["demo"].splitlines():
            w(f"  |   {line}")
        w(f"  | passing tests: {len(payload['passing'])}")
        show_proposed(payload, w)
        w("  > yes | <what's wrong or missing; becomes a new Behavior> | abort")
    else:
        for c in payload["changes"]:
            flags = f"  [{', '.join(c['flags'])}]" if c["flags"] else ""
            w(f"  | {c['status']:<8} {c['path']}{flags}")
        w(payload["diff"])
        w(f"  Workspace: {payload['workspace']}")
        w("  > yes (apply the diff to the project) | no (keep the Workspace)")
    out.flush()


def show_proposed(payload: dict[str, Any], w: Callable[[str], None]) -> None:
    """F10: the Invariants the Arbiter proposed. Never added by the harness: the human adds them to the Plan, or not."""
    for p in payload.get("proposed_invariants", []):
        w(f"  | proposed Invariant (from the Arbiter, on {p['test']}; not applied): {p['text']}: `{p['check']}`"
          + (f" (for {p['fn']})" if p.get("fn") else ""))


@dataclass(frozen=True)
class BuildResult:
    code: int
    summary: Path | None  # None when the run never started (uv sync failed)
    workspace: Workspace
    cost_usd: float
    status: str | None


async def run_build(
    plan: Plan, project: Path, *, model: WorkflowModel, model_id: str, sandbox: Sandbox, config: HarnessConfig,
    hints: bool, ask: Ask, out: TextIO = sys.stdout, err: TextIO = sys.stderr, clock: Callable[[], float] = time.monotonic,
    tty: bool | None = None, plan_path: Path | None = None, backend: str | None = None,
) -> int:
    result = await build_once(
        plan, project, model=model, model_id=model_id, sandbox=sandbox, config=config, hints=hints, ask=ask, out=out,
        err=err, clock=clock, tty=tty, plan_path=plan_path, backend=backend,
    )
    return result.code


async def build_once(
    plan: Plan, project: Path, *, model: WorkflowModel, model_id: str, sandbox: Sandbox, config: HarnessConfig,
    hints: bool, ask: Ask, out: TextIO = sys.stdout, err: TextIO = sys.stderr, clock: Callable[[], float] = time.monotonic,
    tty: bool | None = None, plan_path: Path | None = None, backend: str | None = None, bench_label: str | None = None,
    run_id: str | None = None, arbiter: WorkflowModel | None = None,
) -> BuildResult:
    """One Execution loop run; its summary carries the provenance (F2). ``backend`` defaults to the model id's.
    ``arbiter`` defaults to the one ``[workflow] arbiter`` names (F10), if any."""
    cfg = config.workflow
    arbiter = arbiter if arbiter is not None else make_arbiter(config)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}"
    backend = backend or choose(config, model=model_id).backend
    progress = Progress(err, out, clock=clock, tty=tty)
    rec = Recorder()
    log = RunLog(cfg.runs_dir.expanduser() / f"{run_id}.jsonl", listener=lambda record: progress.end(describe(record)))
    workspace = await Workspace.create(project, cfg.workspaces_dir.expanduser(), run_id)
    out.write(f"plan: {plan.goal}\nmodel: {model_id}\nworkspace: {workspace.path}\nrun log: {log.path}\n")
    if arbiter is not None:
        out.write(f"arbiter: {cfg.arbiter}, at most {cfg.max_arbitrations} per Behavior\n")
    if workspace.warning:
        out.write(f"warning: {workspace.warning}\n")
    synced = await workspace.sync_deps()
    if synced.exit_code != 0:
        err.write(f"uv sync --frozen failed:\n{synced.output}\n")
        return BuildResult(1, None, workspace, 0.0, None)
    graph = build_execution_graph(model, sandbox, workspace, cfg, hints=hints, recorder=rec, log=log, arbiter=arbiter)
    seen ={"idx": 0, "total": len(plan.behaviors)}

    def on_start(node: str) -> None:
        progress.set_behavior(seen["idx"], seen["total"])
        progress.begin(node)

    def on_update(node: str, update: dict[str, Any]) -> None:
        if "idx" in update:
            seen["idx"] = update["idx"]
        if "plan" in update:
            seen["total"] = len(update["plan"]["behaviors"])

    async def answer(payload: dict[str, Any]) -> str:
        progress.pause()
        show(payload, out)
        if payload["kind"] != "finish" and not hints:
            raise RuntimeError(f"unexpected {payload['kind']} interrupt in a no-hints run")
        result = ask(payload)
        return str(await result) if asyncio.iscoroutine(result) else str(result)

    ticker = asyncio.ensure_future(progress.ticker())
    started = clock()
    cancelled = False
    try:
        final = await drive_graph(graph, initial_state(plan), answer, on_start=on_start, on_update=on_update)
    except asyncio.CancelledError:
        # Ctrl-C: the cancellation already closed the in-flight HTTP stream and killed any Sandbox process group
        cancelled = True
        final = {"status": "cancelled", "plan": {"behaviors": plan.behaviors}}
    finally:
        ticker.cancel()
        progress.clear()
    summary = write_summary(
        project / "runs", model=model_id, project=project, no_hints=not hints, status=final.get("status"),
        planned=len(final["plan"]["behaviors"]), rec=rec, workspace=workspace.path, run_log=log.path, name=run_id,
        provenance=bench.provenance(
            plan, plan_path=plan_path, backend=backend, effort=cfg.effort if backend == "anthropic" else None,
            max_attempts=cfg.max_attempts, execution=execution_variant(cfg), bench=bench_label,
        ),
    )
    status = final.get("status")
    if cancelled:
        out.write(f"\ncancelled; Workspace kept at {workspace.path}\nsummary: {summary}\n")
        out.flush()
        return BuildResult(CANCELLED_EXIT, summary, workspace, rec.cost_usd, status)
    out.write(final_report(rec, clock() - started) + f"\n\nstatus: {status}; polish {rec.polish}\nsummary: {summary}\n")
    if status == "api_error":
        out.write(api_error_note(final["api_error"]) + f"; Workspace kept at {workspace.path}\n")
    out.flush()
    code = SPEND_CAP_EXIT if status == "spend_cap" else API_ERROR_EXIT if status == "api_error" else 0
    return BuildResult(code, summary, workspace, rec.cost_usd, status)


@dataclass(frozen=True)
class Cell:
    """One point of a bench grid (F12): a model at an effort, under its own label. A plain bench is one Cell."""

    label: str
    choice: Choice
    config: HarnessConfig  # the bench's config, with this Cell's effort
    make: Callable[[], WorkflowModel]  # a fresh model per run


@dataclass(frozen=True)
class BenchRun:
    cell: Cell
    repeat: int
    pair: bench.Pair


@dataclass(frozen=True)
class RunOutcome:
    code: int
    status: str | None
    cost_usd: float
    capped: bool  # stopped by a Spend cap the bench-wide max_cost had lowered: the bench has spent what it may


async def run_bench(
    bench_set: bench.BenchSet, *, make: Callable[[], WorkflowModel], choice: Choice, sandbox: Sandbox, config: HarnessConfig,
    repeat: int, label: str, max_cost: float, ledger: Path, out: TextIO = sys.stdout, err: TextIO = sys.stderr,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Every pair ``repeat`` times, ``--no-hints`` and answering ``no`` at finish, one ledger line per run; each Workspace
    is removed afterwards. Each run's Spend cap is the smaller of ``max_cost_usd`` and what is left of ``max_cost``; once
    the bench has spent ``max_cost`` it stops cleanly (exit 3), keeping the lines already written. A pair with Hidden
    tests (F7) is graded before its Workspace is removed, and the grade goes into its summary and ledger line."""
    return await run_grid(
        bench_set, [Cell(label, choice, config, make)], sandbox=sandbox, repeat=repeat, max_cost=max_cost, ledger=ledger,
        out=out, err=err, clock=clock,
    )


async def run_grid(
    bench_set: bench.BenchSet, cells: list[Cell], *, sandbox: Sandbox, repeat: int, max_cost: float, ledger: Path,
    parallel: int = 1, out: TextIO = sys.stdout, err: TextIO = sys.stderr, clock: Callable[[], float] = time.monotonic,
) -> int:
    """``run_bench`` over every Cell of a grid (F12), each under its own label, with one ``max_cost`` for the whole grid.

    With ``parallel`` above 1, that many runs go at once (a batched bench needs them all waiting together). The
    bench-wide ``max_cost`` still holds: each run starts with a Spend cap of the smaller of ``max_cost_usd`` and what is
    left once the caps of the runs already going are set aside, and a run that would get none waits for one to end."""
    runs = [BenchRun(cell, n, pair) for cell in cells for n in range(1, repeat + 1) for pair in bench_set.pairs]
    if len(cells) == 1:
        c = cells[0]
        out.write(f"bench {bench_set.name} on {c.choice.model_id}: {len(bench_set.pairs)} pairs x {repeat} = {len(runs)} "
                  f"runs, label {c.label}, max ${max_cost:.2f}, ledger {ledger}\n")
    else:
        out.write(f"bench {bench_set.name}: {len(cells)} cells x {len(bench_set.pairs)} pairs x {repeat} = {len(runs)} runs, "
                  f"max ${max_cost:.2f}, ledger {ledger}\n")
        for c in cells:
            out.write(f"  {c.label}: {c.choice.model_id}, effort {c.config.workflow.effort or 'default'}\n")
    if parallel > 1:
        out.write(f"{min(parallel, len(runs))} runs at a time\n")
    named = len(cells) > 1

    async def one(run: BenchRun, cap: float) -> RunOutcome | None:
        return await bench_one(bench_set, run, cap=cap, repeat=repeat, sandbox=sandbox, ledger=ledger, out=out, err=err,
                               clock=clock, named=named)

    def stopped(spent: float, done: int) -> int:
        out.write(f"bench stopped at ${spent:.4f} (max ${max_cost:.2f}) after {done}/{len(runs)} runs\n")
        return SPEND_CAP_EXIT

    if parallel <= 1:
        spent, done = 0.0, 0
        for run in runs:
            left = max_cost - spent
            if left <= 0:
                return stopped(spent, done)
            outcome = await one(run, min(run.cell.config.workflow.max_cost_usd, left))
            if outcome is None:
                continue
            spent += outcome.cost_usd
            done += 1
            if outcome.code == CANCELLED_EXIT:
                return CANCELLED_EXIT
            if outcome.capped:
                return stopped(spent, done)
        out.write(f"bench done: {done} runs, ${spent:.4f}\n")
        return 0

    queue = list(runs)
    tally: dict[str, float] = {"spent": 0.0, "reserved": 0.0, "running": 0, "done": 0}
    stop: list[int] = []  # the exit code that stops the bench from starting more runs
    turn = asyncio.Condition()

    async def worker() -> None:
        while True:
            async with turn:
                while True:
                    if stop or not queue:
                        return
                    left = max_cost - tally["spent"] - tally["reserved"]
                    if left > 0:
                        break
                    if not tally["running"]:
                        stop.append(SPEND_CAP_EXIT)
                        return
                    await turn.wait()
                run = queue.pop(0)
                cap = min(run.cell.config.workflow.max_cost_usd, left)
                tally["reserved"] += cap
                tally["running"] += 1
            outcome: RunOutcome | None = None
            try:
                outcome = await one(run, cap)
            finally:
                async with turn:
                    tally["reserved"] -= cap
                    tally["running"] -= 1
                    if outcome is not None:
                        tally["spent"] += outcome.cost_usd
                        tally["done"] += 1
                        if outcome.code == CANCELLED_EXIT or outcome.capped:
                            stop.append(CANCELLED_EXIT if outcome.code == CANCELLED_EXIT else SPEND_CAP_EXIT)
                    turn.notify_all()

    await asyncio.gather(*(worker() for _ in range(min(parallel, len(runs)))))
    spent, done = tally["spent"], int(tally["done"])
    if CANCELLED_EXIT in stop:
        return CANCELLED_EXIT
    if stop:
        return stopped(spent, done)
    out.write(f"bench done: {done} runs, ${spent:.4f}\n")
    return 0


async def bench_one(
    bench_set: bench.BenchSet, run: BenchRun, *, cap: float, repeat: int, sandbox: Sandbox, ledger: Path, out: TextIO,
    err: TextIO, clock: Callable[[], float], named: bool = False,
) -> RunOutcome | None:
    """One bench run under Spend cap ``cap``: build, grade (F7), one ledger line, remove the Workspace. None when the
    run never started. A batched model (F12) is closed as soon as the run ends, so the others stop waiting for it."""
    cell, pair, n = run.cell, run.pair, run.repeat
    run_config = cell.config.model_copy(deep=True)
    run_config.workflow.max_cost_usd = cap
    plan = Plan.load(pair.plan)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = re.sub(r"[^A-Za-z0-9._-]", "_", f"{stamp}-{cell.choice.model_id}-{cell.label}-{pair.key}-r{n}")
    started = clock()
    log_out = io.StringIO()  # the run's own output; the bench prints one line per run instead
    model = cell.make()
    try:
        result = await build_once(
            plan, pair.project, model=model, model_id=cell.choice.model_id, sandbox=sandbox, config=run_config, hints=False,
            ask=lambda _p: "no", out=log_out, err=log_out, clock=clock, tty=False, plan_path=pair.plan,
            backend=cell.choice.backend, bench_label=cell.label, run_id=run_id, arbiter=batched_arbiter(model, run_config),
        )
    finally:
        if isinstance(model, BatchedWorkflowModel):
            model.close()
    prefix = f"[{cell.label}] " if named else ""
    try:
        if result.summary is None:
            err.write(f"{prefix}{pair.key} r{n}: the run did not start\n{log_out.getvalue()}")
            return None
        summary = json.loads(result.summary.read_text())
        if pair.hidden is not None and result.code != CANCELLED_EXIT:
            graded = await bench.grade(
                result.workspace, pair.hidden, pair.module, sandbox=sandbox,
                test_command=run_config.workflow.test_command, timeout_s=run_config.workflow.command_timeout_s,
            )
            summary = amend_summary(result.summary, graded.fields())
        bench.append_ledger(ledger, bench.ledger_line(label=cell.label, set_name=bench_set.name, repeat=n, pair=pair, summary=summary))
    finally:
        await result.workspace.remove()
    t = summary["totals"]
    out.write(f"{prefix}{pair.key} r{n}/{repeat}: {t['green']}/{summary['planned']} Green · {t['attempts']}a · "
              f"{t.get('satisfied', 0)}s · {summary.get('stuck', 0)} stuck · {(clock() - started) / 60:.1f}m · "
              f"${result.cost_usd:.4f} · {result.status}{hidden_note(summary)}\n")
    out.flush()
    capped = result.status == "spend_cap" and cap < cell.config.workflow.max_cost_usd
    return RunOutcome(result.code, result.status, result.cost_usd, capped)


def batched_arbiter(model: WorkflowModel, config: HarnessConfig) -> WorkflowModel | None:
    """A batched run's Arbiter (F10) goes through the run's own lane, so its calls join the same batches, and without
    server-side fallbacks, which Batches reject. None for an unbatched run: ``build_once`` makes the usual one."""
    role = arbiter_role(config.workflow)
    if role is None or not isinstance(model, BatchedWorkflowModel):
        return None
    return BatchedWorkflowModel(model.lane, role.model, effort=role.effort)


def hidden_note(summary: dict[str, Any]) -> str:
    """F7: the hidden-test grade on a bench line, when the run was graded."""
    if "hidden_total" not in summary:
        return ""
    return (f" · hidden {summary['hidden_passed']}/{summary['hidden_total']} · "
            f"{summary['mutants_killed']}/{summary['mutants_total']} mutants killed")


def api_error_note(error: dict[str, Any]) -> str:
    """F8: where the run stopped and why, for a human."""
    status = f" {error['status_code']}" if error.get("status_code") else ""
    return (f"stopped by an API error in {error['node']}: {error['error']}{status} "
            f"(request {error.get('request_id') or 'unknown'}); nothing was applied")


def cost_note(rec: Recorder) -> str:
    return f", ${rec.cost_usd:.4f}" if rec.cost_usd else ""


def plan_id_for(goal: str, stamp: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:40].strip("-")
    return f"{stamp}-{slug or 'plan'}"


async def run_plan(
    goal: str, project: Path, *, model: WorkflowModel, model_id: str, config: HarnessConfig, ask: Ask,
    out: TextIO = sys.stdout, err: TextIO = sys.stderr, clock: Callable[[], float] = time.monotonic, tty: bool | None = None,
) -> int:
    """Run the Plan loop on ``goal``, researching ``project``; prints the Plan as Markdown and the path it was saved to."""
    cfg = config.workflow
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plan_id = plan_id_for(goal, stamp)
    progress = Progress(err, out, clock=clock, tty=tty, labelled=PLAN_NODES)
    progress.set_round(1, cfg.max_rounds)
    rec = Recorder()
    log = RunLog(
        cfg.runs_dir.expanduser() / f"plan-{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}.jsonl",
        listener=lambda record: progress.end(describe(record)),
    )
    out.write(f"goal: {goal}\nproject: {project}\nmodel: {model_id}\nrun log: {log.path}\n")
    graph = build_plan_graph(model, project, cfg, plan_id=plan_id, recorder=rec, log=log)

    def on_update(node: str, update: dict[str, Any]) -> None:
        if "round" in update:
            progress.set_round(update["round"], cfg.max_rounds)

    async def answer(payload: dict[str, Any]) -> str:
        progress.pause()
        show(payload, out)
        result = ask(payload)
        return str(await result) if asyncio.iscoroutine(result) else str(result)

    ticker = asyncio.ensure_future(progress.ticker())
    started = clock()
    try:
        final = await drive_graph(graph, initial_plan_state(goal), answer, on_start=progress.begin, on_update=on_update, thread_id="plan")
    except asyncio.CancelledError:
        out.write(f"\ncancelled; no Plan saved\nrun log: {log.path}\n")
        out.flush()
        return CANCELLED_EXIT
    except WorkflowModelError as exc:
        err.write(f"\nthe model failed: {exc}\nrun log: {log.path}\n")
        return 1
    finally:
        ticker.cancel()
        progress.clear()
    if final.get("status") == "api_error":
        out.write(f"\n{api_error_note(final['api_error'])}; no Plan saved\nrun log: {log.path}\n")
        out.flush()
        return API_ERROR_EXIT
    if final.get("status") == "spend_cap":
        out.write(f"\nstopped at the Spend cap (${rec.cost_usd:.4f} > ${cfg.max_cost_usd:.2f}); no Plan saved\nrun log: {log.path}\n")
        out.flush()
        return SPEND_CAP_EXIT
    path = final["plan_path"]
    out.write(
        f"\n{final['markdown']}\ntotal wall time {clock() - started:.1f}s, model time {rec.model_seconds:.1f}s over "
        f"{rec.model_calls} calls{cost_note(rec)}, {final['round']} round(s), status: {final['status']}\n"
        f"plan saved: {path}\nbuild it: harness-workflow build --plan {path} --project {project}\n"
    )
    out.flush()
    return 0


async def with_model(
    args: argparse.Namespace, config: HarnessConfig, body: Callable[[WorkflowModel, str], Awaitable[int]],
) -> int:
    """Build the model for ``--backend``/``--model``/config, then run ``body`` with it. An Ollama model is preflighted and
    pinned the way ``evals/runner.py`` does; an API run skips that, the Guard has nothing to watch (ADR 0006)."""
    choice = choose(config, model=args.model, backend=args.backend)
    if choice.backend == "anthropic":
        model = make_model(config, choice, None)
        assert model is not None
        sys.stdout.write(f"model {choice.model_id} on the Anthropic API; spend cap ${config.workflow.max_cost_usd:.2f}\n")
        return await body(model, choice.model_id)
    model_id = choice.model_id
    client = OllamaClient(config.ollama.host, keep_alive="10m")  # every request pins num_ctx, so models never Reload
    try:
        try:
            pre = await run_preflight(client, [Candidate(model_id, config.workflow.num_ctx)], env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            sys.stderr.write(f"preflight failed: {exc}\n")
            return 1
        sys.stdout.write(f"model {model_id} pinned at num_ctx {pre.num_ctx}\n")
        model = make_model(config, choice, client)
        assert model is not None
        return await body(model, model_id)
    finally:
        if args.unload:
            await client.unload(model_id)
        await client.aclose()


async def live_build(args: argparse.Namespace, config: HarnessConfig, ask: Ask) -> int:
    plan = Plan.load(args.plan)
    project = args.project.expanduser().resolve()

    async def body(model: WorkflowModel, model_id: str) -> int:
        return await run_build(
            plan, project, model=model, model_id=model_id, sandbox=SeatbeltSandbox(), config=config, hints=not args.no_hints, ask=ask,
            plan_path=args.plan.expanduser().resolve(),
        )

    return await with_model(args, config, body)


async def live_bench(args: argparse.Namespace, config: HarnessConfig) -> int:
    sets = bench.load_manifest(args.manifest)
    if args.set not in sets:
        sys.stderr.write(f"no set {args.set!r} in {args.manifest} (sets: {', '.join(sets)})\n")
        return 2
    if args.no_red_check:
        config.workflow.check_tests_at_red = False
    if args.hide_rules:
        config.workflow.show_invariants = False
    batcher = Batcher(make_client(config.anthropic), flush_s=args.flush_s) if args.batch else None

    def maker(cell_config: HarnessConfig, choice: Choice) -> Callable[[], WorkflowModel]:
        def make() -> WorkflowModel:
            if batcher is not None:
                return BatchedWorkflowModel(batcher.lane(), choice.model_id, effort=cell_config.workflow.effort)
            model = make_model(cell_config, choice, None)
            assert model is not None
            return model

        return make

    try:
        cells = grid_cells(args, config, maker)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    bench_set = sets[args.set]
    parallel = args.parallel or (len(cells) * len(bench_set.pairs) * args.repeat if args.batch else 1)
    try:
        return await run_grid(
            bench_set, cells, sandbox=SeatbeltSandbox(), repeat=args.repeat, max_cost=args.max_cost,
            ledger=config.paths.bench_ledger, parallel=parallel,
        )
    finally:
        if batcher is not None:
            await batcher.aclose()


def grid_cells(
    args: argparse.Namespace, config: HarnessConfig,
    maker: Callable[[HarnessConfig, Choice], Callable[[], WorkflowModel]],
) -> list[Cell]:
    """The bench's Cells: one for ``--model`` (labelled ``--label``), or with ``--models``/``--effort`` (F12) one per model
    x effort, labelled ``<label>-<model>-<effort>``. Raises ValueError for an Ollama model (ADR 0007), an unknown effort,
    or a model with no price, before anything runs."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    grid = bool(args.models or args.effort)
    models: list[str | None] = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [args.model]
    efforts: list[Effort | None] = [config.workflow.effort]
    if args.effort:
        known = get_args(Effort)
        asked = [e.strip() for e in args.effort.split(",") if e.strip()]
        if bad := [e for e in asked if e not in known]:
            raise ValueError(f"unknown effort {', '.join(bad)} (known: {', '.join(known)})")
        efforts = [cast(Effort, e) for e in asked]
    if role := arbiter_role(config.workflow):
        require_pricing(role.model)  # F10: the Arbiter's calls count toward the Spend cap, so it needs a price too
    cells: list[Cell] = []
    for model in models:
        choice = choose(config, model=model, backend=args.backend)
        if choice.backend != "anthropic":
            raise ValueError("bench runs on the Anthropic API; the Ollama path is frozen (ADR 0007)")
        require_pricing(choice.model_id)  # a sweep fails here, not one cell in
        for effort in efforts:
            cell_config = config.model_copy(deep=True)
            cell_config.workflow.effort = effort
            if grid:
                label = f"{args.label or f'{args.set}-{stamp}'}-{choice.model_id}-{effort or 'default'}"
            else:
                label = args.label or f"{args.set}-{choice.model_id}-{stamp}"
            cells.append(Cell(re.sub(r"[^A-Za-z0-9._-]", "_", label), choice, cell_config, maker(cell_config, choice)))
    return cells


def report(args: argparse.Namespace, config: HarnessConfig, out: TextIO = sys.stdout) -> int:
    if args.summaries is not None:
        runs = bench.load_summaries(args.summaries.expanduser(), args.all)
        out.write((bench.summaries_table(runs) if runs else "no runs found") + "\n")
        return 0
    ledger = args.ledger or config.paths.bench_ledger
    lines = bench.read_ledger(ledger)
    if args.frontier:
        out.write(bench.frontier_table(lines, args.frontier) + "\n")
    elif args.compare:
        out.write(bench.compare_table(lines, *args.compare) + "\n")
    elif args.labels:
        out.write("\n\n".join(bench.label_table(lines, label) for label in args.labels) + "\n")
    else:
        rows = bench.labels(lines)
        out.write(("labels in " + str(ledger) + ":\n" + "\n".join(f"  {name}  {n} runs  from {at}" for name, n, at in rows))
                  if rows else f"the ledger {ledger} is empty")
        out.write("\n")
    return 0


async def live_plan(args: argparse.Namespace, config: HarnessConfig, ask: Ask) -> int:
    project = args.project.expanduser().resolve()

    async def body(model: WorkflowModel, model_id: str) -> int:
        return await run_plan(args.goal, project, model=model, model_id=model_id, config=config, ask=ask)

    return await with_model(args, config, body)


async def _stdin(_payload: dict[str, Any]) -> str:
    return await asyncio.to_thread(input, "  > ")


async def run_cancellable(coro: Coroutine[Any, Any, int]) -> int:
    """Run ``coro`` as a task that the first SIGINT cancels; a second SIGINT falls back to KeyboardInterrupt."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(coro)

    def interrupt() -> None:
        loop.remove_signal_handler(signal.SIGINT)
        task.cancel()

    loop.add_signal_handler(signal.SIGINT, interrupt)
    try:
        return await task
    except asyncio.CancelledError:
        return CANCELLED_EXIT
    finally:
        loop.remove_signal_handler(signal.SIGINT)


def main(argv: list[str] | None = None, *, ask: Ask = _stdin) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "report":
        code = report(args, config)
    elif args.command == "bench":
        code = asyncio.run(run_cancellable(live_bench(args, config)))
    else:
        if args.max_cost is not None:
            config.workflow.max_cost_usd = args.max_cost
        run = live_plan if args.command == "plan" else live_build
        code = asyncio.run(run_cancellable(run(args, config, ask)))
    if code:
        raise SystemExit(code)
