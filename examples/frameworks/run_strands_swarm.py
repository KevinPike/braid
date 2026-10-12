"""F13 runner 2: a Strands ``Swarm``, a tester agent and an implementer agent that hand off to each other.

    uv run --env-file .env python examples/frameworks/run_strands_swarm.py --kata ferry-fare --out runs/strands
    uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/strands

Strands' own Anthropic provider pins ``anthropic<1.0``, so each agent runs on braid's ``AnthropicChatModel``
(``harness/agent.py``): Strands' formatting, a request built and sent on the 1.x client, no sampling parameters, and
every call priced on a ``UsageMeter`` (one per Role). Routing is emergent: the Swarm gives each agent a
``handoff_to_agent`` tool and the model decides when to use it, bounded by ``max_handoffs`` and ``max_iterations``.

Isolation: model-written code runs here, outside braid's Sandbox and ADR 0005, in a temporary copy of the kata
(``common.py``). The tools are this file's: file access is confined to the copy, each Role writes only its own files,
and pytest runs without any credential in its environment. The key is in this process (the client needs it) but no tool
passes it on. The trace is ``<out>/.trace/swarm.json``: the handoff order and each agent's messages.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

from common import (DEFAULT_MODEL, TASK, Role, ToolRefused, Usage, export, implementer_prompt, kata_copy, list_files,
                    read_file, report, run_tests, tester_prompt, write_file, write_trace)
from strands import Agent, tool
from strands.multiagent import Swarm

from harness.agent import AnthropicChatModel
from harness.anthropic_api import make_client
from harness.config import AnthropicConfig
from harness.usage import CONTEXT_WINDOWS, UsageMeter

MAX_HANDOFFS = 6  # tester -> implementer, and at most a couple of disputes back and forth
MAX_ITERATIONS = 8
NODE_TIMEOUT_S = 600.0
EXECUTION_TIMEOUT_S = 1800.0


def tools_for(work: Path, role: Role) -> list[Any]:
    """The four tools, bound to the kata copy and to ``role``'s write rule."""

    def guarded(fn: Any, *args: Any) -> str:
        try:
            return str(fn(*args))
        except ToolRefused as exc:
            return f"refused: {exc}"

    @tool
    def read(path: str) -> str:
        """Read a file in the kata directory, by its relative path (for example GOAL.md)."""
        return guarded(read_file, work, path)

    @tool
    def write(path: str, content: str) -> str:
        """Write a whole file in the kata directory, by its relative path, replacing any old content."""
        return guarded(write_file, work, role, path, content)

    @tool
    def ls() -> str:
        """List the files in the kata directory."""
        return list_files(work)

    @tool
    def pytest(path: str = "") -> str:
        """Run pytest in the kata directory (optionally on one test file) and return its exit code and output."""
        return guarded(run_tests, work, path)

    return [read, write, ls, pytest]


def agent(work: Path, role: Role, model_id: str, meter: UsageMeter, prompt: str, description: str) -> Agent:
    model = AnthropicChatModel(make_client(AnthropicConfig()), model_id, meter=meter)
    # callback_handler=None: stdout carries only the JSON line. retry_strategy=None: the SDK's retries are the only ones.
    return Agent(model=model, name=role, description=description, system_prompt=prompt, tools=tools_for(work, role),
                 callback_handler=None, retry_strategy=None)


async def run(kata: str, out: Path, model_id: str) -> Usage:
    usage = Usage("strands-swarm", kata, model_id, str(out))
    meters = {role: UsageMeter(model_id, CONTEXT_WINDOWS.get(model_id, 200_000)) for role in ("tester", "implementer")}
    started = time.monotonic()
    with kata_copy(kata) as work:
        tester = agent(work, "tester", model_id, meters["tester"], tester_prompt(kata),
                       "Writes pytest tests from GOAL.md; never writes the implementation.")
        implementer = agent(work, "implementer", model_id, meters["implementer"], implementer_prompt(kata),
                            "Writes the module so the tester's tests pass; never edits tests.")
        swarm = Swarm([tester, implementer], entry_point=tester, max_handoffs=MAX_HANDOFFS, max_iterations=MAX_ITERATIONS,
                      execution_timeout=EXECUTION_TIMEOUT_S, node_timeout=NODE_TIMEOUT_S)
        try:
            result = await swarm.invoke_async(TASK)
            usage.status = str(result.status.value if hasattr(result.status, "value") else result.status)
            history = [node.node_id for node in result.node_history]
        except Exception as exc:  # a failed run still reports what it spent, and keeps what it wrote
            usage.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
            history = []
        export(work, out)
        usage.trace = str(write_trace(out, "swarm.json", {
            "handoffs": history,
            "messages": {a.name: a.messages for a in (tester, implementer)},
        }))
    usage.seconds = round(time.monotonic() - started, 1)
    for role, meter in meters.items():
        calls = meter.calls
        usage.roles[role] = {"calls": len(calls), "cost_usd": round(sum(c.cost_usd for c in calls), 6)}
        usage.calls += len(calls)
        usage.input_tokens += sum(c.input_tokens for c in calls)
        usage.output_tokens += sum(c.output_tokens for c in calls)
        usage.cache_read_input_tokens += sum(c.cache_read_input_tokens for c in calls)
        usage.cache_creation_input_tokens += sum(c.cache_creation_input_tokens for c in calls)
    usage.cost_usd = sum(m.cost_usd for m in meters.values())
    usage.cost_source = "braid's UsageMeter on AnthropicChatModel (per call, from each reply's usage)"
    usage.notes = [f"handoffs: {' -> '.join(history) or '-'}"]
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description="F13: a Strands Swarm (tester, implementer) on one exotic kata")
    parser.add_argument("--kata", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    report(asyncio.run(run(args.kata, args.out.resolve(), args.model)))


if __name__ == "__main__":
    main()
