"""``harness-workflow build``: run the Execution loop on a Plan in a Workspace of a git project (M13)."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from harness.config import HarnessConfig, load_config
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, run_preflight, system_gpu_limit
from harness.sandbox import Sandbox, SeatbeltSandbox, Workspace
from harness.workflow.execute import Ask, build_execution_graph, drive_graph, initial_state
from harness.workflow.model import GuardedWorkflowModel, WorkflowModel
from harness.workflow.plan import Plan
from harness.workflow.progress import Progress, describe, final_report
from harness.workflow.records import Recorder, RunLog, write_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness-workflow", description="Run a Plan through the Execution loop.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="make each Behavior of a Plan Red then Green in a sandboxed Workspace")
    b.add_argument("--plan", type=Path, required=True, help="Plan JSON")
    b.add_argument("--project", type=Path, default=Path("."), help="the git project to build in (default .)")
    b.add_argument("--model", help="model id (default [workflow] model, then the [profile] model)")
    b.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    b.add_argument("--no-hints", action="store_true", help="baseline: Stuck skips the Behavior, clarify waives, Sign-off says yes")
    b.add_argument("--unload", action="store_true", help="unload the model at the end")
    return p


def show(payload: dict[str, Any], out: TextIO) -> None:
    def w(text: str = "") -> None:
        out.write(text + "\n")

    w(f"\n  HUMAN: {payload['kind']}")
    if payload["kind"] == "clarify":
        for f in payload["failed"]:
            w(f"  | assumption failed: {f}")
        for q in payload["open_questions"]:
            w(f"  | open question:     {q}")
        w("  > <how to proceed> | abort")
    elif payload["kind"] == "stuck":
        w(f"  | stuck on: {payload['behavior']}")
        for line in payload["last_run"].splitlines()[-8:]:
            w(f"  |   {line}")
        w("  > <a hint for the model> | skip | abort")
    elif payload["kind"] == "sign_off":
        w("  | demo output:")
        for line in payload["demo"].splitlines():
            w(f"  |   {line}")
        w(f"  | passing tests: {len(payload['passing'])}")
        w("  > yes | <what's wrong or missing; becomes a new Behavior> | abort")
    else:
        for c in payload["changes"]:
            flags = f"  [{', '.join(c['flags'])}]" if c["flags"] else ""
            w(f"  | {c['status']:<8} {c['path']}{flags}")
        w(payload["diff"])
        w(f"  Workspace: {payload['workspace']}")
        w("  > yes (apply the diff to the project) | no (keep the Workspace)")
    out.flush()


async def run_build(
    plan: Plan, project: Path, *, model: WorkflowModel, model_id: str, sandbox: Sandbox, config: HarnessConfig,
    hints: bool, ask: Ask, out: TextIO = sys.stdout, err: TextIO = sys.stderr, clock: Callable[[], float] = time.monotonic,
    tty: bool | None = None,
) -> int:
    cfg = config.workflow
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}"
    progress = Progress(err, out, clock=clock, tty=tty)
    rec = Recorder()
    log = RunLog(cfg.runs_dir.expanduser() / f"{run_id}.jsonl", listener=lambda record: progress.end(describe(record)))
    workspace = await Workspace.create(project, cfg.workspaces_dir.expanduser(), run_id)
    out.write(f"plan: {plan.goal}\nmodel: {model_id}\nworkspace: {workspace.path}\nrun log: {log.path}\n")
    if workspace.warning:
        out.write(f"warning: {workspace.warning}\n")
    synced = await workspace.sync_deps()
    if synced.exit_code != 0:
        err.write(f"uv sync --frozen failed:\n{synced.output}\n")
        return 1
    graph = build_execution_graph(model, sandbox, workspace, cfg, hints=hints, recorder=rec, log=log)
    seen = {"idx": 0, "total": len(plan.behaviors)}

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
    try:
        final = await drive_graph(graph, initial_state(plan), answer, on_start=on_start, on_update=on_update)
    finally:
        ticker.cancel()
        progress.clear()
    summary = write_summary(
        project / "runs", model=model_id, project=project, no_hints=not hints, status=final.get("status"),
        planned=len(final["plan"]["behaviors"]), rec=rec, workspace=workspace.path, run_log=log.path,
    )
    out.write(final_report(rec, clock() - started) + f"\n\nstatus: {final.get('status')}; polish {rec.polish}\nsummary: {summary}\n")
    out.flush()
    return 0


async def live_build(args: argparse.Namespace, config: HarnessConfig, ask: Ask) -> int:
    plan = Plan.load(args.plan)
    project = args.project.expanduser().resolve()
    model_id = args.model or config.workflow.model or config.profile.model
    client = OllamaClient(config.ollama.host, keep_alive="10m")  # every request pins num_ctx, so models never Reload
    try:
        try:
            pre = await run_preflight(client, [Candidate(model_id, config.workflow.num_ctx)], env=os.environ, gpu_limit=system_gpu_limit())
        except PreflightError as exc:
            sys.stderr.write(f"preflight failed: {exc}\n")
            return 1
        sys.stdout.write(f"model {model_id} pinned at num_ctx {pre.num_ctx}\n")
        return await run_build(
            plan, project, model=GuardedWorkflowModel(client, model_id), model_id=model_id, sandbox=SeatbeltSandbox(),
            config=config, hints=not args.no_hints, ask=ask,
        )
    finally:
        if args.unload:
            await client.unload(model_id)
        await client.aclose()


async def _stdin(_payload: dict[str, Any]) -> str:
    return await asyncio.to_thread(input, "  > ")


def main(argv: list[str] | None = None, *, ask: Ask = _stdin) -> None:
    args = build_parser().parse_args(argv)
    code = asyncio.run(live_build(args, load_config(args.config), ask))
    if code:
        raise SystemExit(code)
