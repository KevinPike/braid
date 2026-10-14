"""F13 runner 1: braid's own Execution loop (a LangGraph Workflow) with F11's Roles, on one exotic kata.

    uv run --env-file .env python examples/frameworks/run_braid.py --kata ferry-fare --out runs/braid
    uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/braid

The tester and the implementer are separate Roles (``[workflow.roles]``): the tester gets ``TESTER_SYSTEM`` and, by
default, the ``interface`` View, so it never sees a function body. The Plan is ``bench/exotic/<kata>/plan.json``, as
``harness-workflow bench exotic`` uses it. The run goes through ``cli.build_once`` exactly as a bench run does
(``--no-hints``, ``no`` at finish), so model-written code runs only in the Sandbox, in a Workspace (ADR 0005).

Unlike runners 2-4 this one needs the kata repo with a ``.venv`` (the Workspace runs ``uv sync --frozen``): it is made
in a temporary directory with ``make_katas.make(..., sync=True)``, from ``GOAL.md`` and ``summary.txt`` only. The final
files are copied to ``--out``; the summary goes to ``<out>/.trace/summary.json`` and the run log stays in
``~/.braid/runs/``. The key stays out of tool environments: the Sandbox's environment is built from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from common import DEFAULT_MODEL, EXOTIC, Usage, export, kata_repo, report, write_trace

from harness.config import RoleConfig, View, load_config
from harness.sandbox import Sandbox, SeatbeltSandbox
from harness.workflow.backend import choose, make_model, make_roles
from harness.workflow.cli import build_once
from harness.workflow.plan import Plan


async def run(kata: str, out: Path, model_id: str, config_path: Path | None, view: View, sandbox: Sandbox) -> Usage:
    config = load_config(config_path)
    roles = config.workflow.roles
    if not roles.separated:  # F11's Roles, whatever braid.toml says; an explicit [workflow.roles] is kept
        roles.tester = RoleConfig(view=view)
    choice = choose(config, model=model_id)
    if choice.backend != "anthropic":
        raise SystemExit(f"{model_id} is not an Anthropic model")
    model = make_model(config, choice, None)
    assert model is not None
    plan = Plan.load(EXOTIC / kata / "plan.json")
    usage = Usage("braid", kata, model_id, str(out))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"braid-f13-{kata}-") as tmp:
        repo = kata_repo(Path(tmp).resolve(), kata, sync=True)
        result = await build_once(
            plan, repo, model=model, model_id=model_id, sandbox=sandbox, config=config, hints=False,
            ask=lambda _: "no", out=sys.stderr, err=sys.stderr, tty=False, plan_path=EXOTIC / kata / "plan.json",
            roles=make_roles(config, model_id),
        )
        try:
            out.mkdir(parents=True, exist_ok=True)
            export(result.workspace.path, out)
            if result.summary is None:
                usage.status = "did not start"
                return usage
            summary = json.loads(result.summary.read_text())
            usage.trace = str(write_trace(out, "summary.json", summary))
        finally:
            await result.workspace.remove()
    t = summary["totals"]
    usage.calls = t["model_calls"]
    usage.output_tokens = t["output_tokens"]
    usage.cache_read_input_tokens = t["cache_read_input_tokens"]
    usage.input_tokens = t["input_tokens"]
    usage.cost_usd = t["cost_usd"]
    usage.cost_source = "braid's UsageMeter (per call, from each reply's usage)"
    usage.roles = summary.get("roles", {})
    usage.status = str(summary.get("status"))
    usage.seconds = round(time.monotonic() - started, 1)
    usage.notes = [f"run log: {summary.get('run_log')}", f"execution: {summary.get('execution')}",
                   f"Green {t['green']}/{summary['planned']}"]
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description="F13: braid's Workflow with F11 Roles on one exotic kata")
    parser.add_argument("--kata", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=None, help="braid.toml (default ./braid.toml if present)")
    parser.add_argument("--tester-view", choices=("interface", "full"), default="interface")
    args = parser.parse_args()
    usage = asyncio.run(run(args.kata, args.out.resolve(), args.model, args.config, args.tester_view, SeatbeltSandbox()))
    report(usage)


if __name__ == "__main__":
    main()
