"""Run the tool-selection eval against each model in a lineup, one model at a time.

Every Ollama request goes through ``OllamaClient`` (the guard client), preflight pins ``num_ctx`` first so
nothing spills or reloads mid-run, and each model is unloaded (keep_alive 0) before the next one loads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import HarnessConfig
from harness.context.ledger import Ledger
from harness.evals.report import CallLog, ModelResult
from harness.guard.backend import Backend
from harness.guard.ollama_client import OllamaClient
from harness.guard.preflight import Candidate, PreflightError, Severity, run_preflight, system_gpu_limit

# The M4 gate's prompt, so eval accuracy is comparable with tests/test_live_tools.py.
EVAL_SYSTEM_PROMPT = "You are a concise assistant. Use a tool only when the request needs one."
# The hand-labelled M4 set stays where the tests read it; this default resolves from the repo checkout.
DEFAULT_CASES = Path(__file__).resolve().parents[3] / "tests" / "data" / "prompts.json"


def load_prompts(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = json.loads(path.read_text())
    return prompts[:limit] if limit is not None else prompts


async def run_model(
    model: str, prompts: list[dict[str, Any]], factory: Callable[[], Any], *, num_ctx: int, log: CallLog,
) -> ModelResult:
    """Score ``model`` on ``prompts``; ``factory`` makes a fresh agent per case and ``log`` collects its calls.

    The seam the unit tests use: a fake factory stands in for Ollama.
    """
    try:
        from harness.evals.scoring import score_model
    except ImportError as exc:
        raise SystemExit(
            f"harness-eval needs Strands Evals ({exc}); run it with `uv run --group evals harness-eval`"
        ) from exc
    result = ModelResult(model, num_ctx=num_ctx)
    result.cases = await score_model(prompts, factory, log)
    return result


@dataclass
class EvalRuntime:
    """What a lineup run needs from the outside world: the guard client, how to build an agent, the machine."""

    backend: Backend
    make_factory: Callable[[str, CallLog], Callable[[], Any]]
    env: Mapping[str, str]
    gpu_limit: int

    async def aclose(self) -> None:
        close = getattr(self.backend, "aclose", None)
        if close is not None:
            await close()


async def run_lineup(
    models: list[str], prompts: list[dict[str, Any]], *, runtime: EvalRuntime, num_ctx: int,
    progress: Callable[[str], None] = lambda _msg: None,
) -> list[ModelResult]:
    pulled = {t.name for t in await runtime.backend.tags()}
    results: list[ModelResult] = []
    for model in models:
        if model not in pulled:
            progress(f"{model}: not pulled, skipped")
            results.append(ModelResult(model, status="skipped", reason="not pulled", num_ctx=num_ctx))
            continue
        progress(f"{model}: preflight at num_ctx {num_ctx}")
        try:
            pre = await run_preflight(runtime.backend, [Candidate(model, num_ctx)], env=runtime.env, gpu_limit=runtime.gpu_limit)
        except PreflightError as exc:
            progress(f"{model}: skipped, {exc}")
            results.append(ModelResult(model, status="skipped", reason=f"preflight: {exc}", num_ctx=num_ctx))
            continue
        try:
            log = CallLog()
            progress(f"{model}: running {len(prompts)} cases")
            result = await run_model(model, prompts, runtime.make_factory(model, log), num_ctx=pre.num_ctx, log=log)
            result.notes = [c.detail for c in pre.checks if c.severity is Severity.WARN]
            progress(f"{model}: {result.correct}/{result.n} correct")
            results.append(result)
        except Exception as exc:  # one model failing must not lose the others' results
            results.append(ModelResult(model, status="failed", reason=f"{type(exc).__name__}: {exc}", num_ctx=num_ctx))
        finally:
            try:
                await runtime.backend.unload(model)  # keep_alive 0 so the next model has the GPU to itself
            except Exception as exc:
                progress(f"{model}: unload failed ({exc})")
    return results


def live_runtime(config: HarnessConfig) -> EvalRuntime:
    """The real thing: the guard client on the configured host, read-only tools bound to the project folder."""
    from harness.agent import build_agent
    from harness.tools.builtin import build_tools

    client = OllamaClient(config.ollama.host, keep_alive="10m")
    current = CallLog()

    def record(_model: str, metrics: Any, _estimate: int | None) -> None:
        current.add(metrics)

    client.on_call(record)

    def make_factory(model: str, log: CallLog) -> Callable[[], Any]:
        nonlocal current
        current = log
        ledger = Ledger(client.pinned)  # estimate vs measured per call; the CallLog keeps the speeds
        tools: list[Any] = list(build_tools(config.tools.root, config.tools.notes_dir, max_read_chars=config.tools.max_read_chars))
        return lambda: build_agent(client, model, EVAL_SYSTEM_PROMPT, ledger, tools=tools)

    return EvalRuntime(client, make_factory, os.environ, system_gpu_limit())
