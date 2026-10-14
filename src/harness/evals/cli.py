"""``harness-eval``: one command that compares the lineup on the M4 tool-selection set."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

from harness.config import HarnessConfig, load_config
from harness.evals.report import write_reports
from harness.evals.runner import DEFAULT_CASES, EvalRuntime, live_runtime, load_prompts, run_lineup


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness-eval",
        description="Run the hand-labelled tool-selection prompts against each model and compare accuracy, "
                    "tokens per second and peak context use. Run with `uv run --group evals harness-eval`.",
    )
    p.add_argument("--config", type=Path, help="path to braid.toml (default ./braid.toml)")
    p.add_argument("--models", help="comma-separated models, overriding [evals] models")
    p.add_argument("--num-ctx", type=int, help="pinned num_ctx for every model (default [evals] num_ctx)")
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES, help=f"prompts JSON (default {DEFAULT_CASES})")
    p.add_argument("--limit", type=int, help="only the first N prompts")
    return p


async def run(
    config: HarnessConfig, args: argparse.Namespace, runtime: EvalRuntime, *, out: Callable[[str], None] = print,
) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list(config.evals.models)
    num_ctx: int = args.num_ctx or config.evals.num_ctx
    prompts = load_prompts(args.cases, args.limit)
    try:
        results = await run_lineup(
            models, prompts, runtime=runtime, num_ctx=num_ctx, progress=lambda m: print(m, file=sys.stderr, flush=True)
        )
    finally:
        await runtime.aclose()
    md_path, json_path, md = write_reports(results, config.paths.evals_dir, num_ctx=num_ctx, cases_path=str(args.cases))
    out(md)
    out(f"saved {md_path} and {json_path}")
    return 0 if any(r.status == "ok" for r in results) else 1


def main(argv: list[str] | None = None, *, runtime_factory: Callable[[HarnessConfig], EvalRuntime] = live_runtime) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    code = asyncio.run(run(config, args, runtime_factory(config)))
    if code:
        raise SystemExit(code)
