"""F13 runner 4: CrewAI, a tester agent and an implementer agent with one task each, run in sequence.

    uv run --env-file .env --with crewai python examples/frameworks/run_crewai.py --kata ferry-fare --out runs/crewai
    uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/crewai

``crewai`` (checked against 1.15.27) is never a project dependency: ``uv run --with`` adds it for one run. Claude is
reached through CrewAI's native Anthropic provider (``LLM(model="anthropic/claude-haiku-5-5")`` resolves to
``AnthropicCompletion``), not LiteLLM. That provider's extra pins ``anthropic~=0.73``; under ``uv run --with crewai``
the project's ``anthropic`` 1.x is the one installed, and an offline probe on a mock transport showed the provider
works on it and sends ``model``, ``messages``, ``max_tokens``, ``stop_sequences`` and ``stream`` only. Its
``temperature`` and ``top_p`` default to None and are sent only when set; this runner checks they are None before
kickoff, since Claude 5.5 models answer 400 to them. CrewAI adds no ``cache_control``, so nothing is cached.

Routing is explicit and linear: ``Process.sequential`` runs the tester's task, then the implementer's with the
tester's answer as context. There is no way back to the tester, so a disputed test cannot be handed back as it can in
the Swarm and in braid.

Isolation: model-written code runs here, outside braid's Sandbox and ADR 0005, in a temporary copy of the kata. The
tools are this file's (``common.py``): file access is confined to the copy, each Role writes only its own files, and
pytest runs without any credential in its environment. The key is in this process (the provider reads
``ANTHROPIC_API_KEY``) but no tool passes it on. CrewAI's telemetry and tracing are turned off. The trace is
``<out>/.trace/crew.json``: every tool call, by Role, and each task's output.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from common import (DEFAULT_MODEL, Role, ToolRefused, Usage, export, implementer_prompt, kata_copy, list_files,
                    no_sampling, price, read_file, report, run_tests, tester_prompt, write_file, write_trace)

for _flag, _value in (("CREWAI_DISABLE_TELEMETRY", "true"), ("OTEL_SDK_DISABLED", "true"),
                      ("CREWAI_TRACING_ENABLED", "false")):
    os.environ.setdefault(_flag, _value)

try:
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool
except ImportError as exc:  # pragma: no cover - only when run without --with crewai
    raise SystemExit("crewai is not installed: run this with `uv run --with crewai ...`") from exc

MAX_ITER = 25  # tool-using steps per agent before CrewAI forces a final answer
MAX_TOKENS = 16000


def tools_for(work: Path, role: Role, log: list[dict[str, Any]]) -> list[Any]:
    """The four tools, bound to the kata copy and to ``role``'s write rule. Each call goes into ``log``: CrewAI's
    native tool calling leaves no step record of its own (``step_callback`` sees only the final answer)."""

    def guarded(fn: Any, *args: Any) -> str:
        try:
            result = str(fn(*args))
        except ToolRefused as exc:
            result = f"refused: {exc}"
        log.append({"role": role, "tool": fn.__name__, "args": [str(a)[:200] for a in args[1:]], "result": result[:1000]})
        return result

    @tool("read_file")
    def read(path: str) -> str:
        """Read a file in the kata directory, by its relative path (for example GOAL.md)."""
        return guarded(read_file, work, path)

    @tool("write_file")
    def write(path: str, content: str) -> str:
        """Write a whole file in the kata directory, by its relative path, replacing any old content."""
        return guarded(write_file, work, role, path, content)

    @tool("list_files")
    def ls() -> str:
        """List the files in the kata directory."""
        return guarded(list_files, work)

    @tool("run_pytest")
    def pytest(path: str = "") -> str:
        """Run pytest in the kata directory (optionally on one test file) and return its exit code and output."""
        return guarded(run_tests, work, path)

    return [read, write, ls, pytest]


def llm(model_id: str) -> Any:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("CrewAI's Anthropic provider needs ANTHROPIC_API_KEY (run with `uv run --env-file .env ...`)")
    made = LLM(model=f"anthropic/{model_id}", max_tokens=MAX_TOKENS)
    if type(made).__name__ != "AnthropicCompletion":
        raise SystemExit(f"CrewAI did not route {model_id} to its native Anthropic provider (got {type(made).__name__})")
    no_sampling({"temperature": made.temperature, "top_p": getattr(made, "top_p", None), "top_k": getattr(made, "top_k", None)})
    return made


def run(kata: str, out: Path, model_id: str) -> Usage:
    usage = Usage("crewai", kata, model_id, str(out))
    llms = {"tester": llm(model_id), "implementer": llm(model_id)}
    calls: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    started = time.monotonic()
    with kata_copy(kata) as work:
        def agent(role: Role, goal: str) -> Agent:
            return Agent(
                role=role, goal=goal, backstory=tester_prompt(kata) if role == "tester" else implementer_prompt(kata),
                llm=llms[role], tools=tools_for(work, role, calls), allow_delegation=False, max_iter=MAX_ITER,
                verbose=False,
            )

        tester = agent("tester", "Pin down every rule in GOAL.md with pytest tests, derived from GOAL.md alone.")
        implementer = agent("implementer", "Make every test pass with a module that meets GOAL.md.")
        write_tests = Task(
            description=tester_prompt(kata) + "\nYour final answer is passed to the implementer.",
            expected_output="The test file's path and the rules its tests cover.", agent=tester,
        )
        implement = Task(
            description=implementer_prompt(kata) + "\nThis crew has no way back to the tester: if a test contradicts "
            "GOAL.md, say so in your final answer and make the code meet GOAL.md.",
            expected_output="Whether every test passes, and any test that contradicts GOAL.md.", agent=implementer,
            context=[write_tests],
        )
        crew = Crew(agents=[tester, implementer], tasks=[write_tests, implement], process=Process.sequential, verbose=False)
        try:
            with contextlib.redirect_stdout(sys.stderr):  # stdout carries only the JSON line
                result = crew.kickoff()
            outputs = [{"agent": str(t.agent), "raw": t.raw} for t in result.tasks_output]
            usage.status = "completed"
        except Exception as exc:  # a failed run still reports what it spent, and keeps what it wrote
            usage.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        export(work, out)
        usage.trace = str(write_trace(out, "crew.json", {"tool_calls": calls, "tasks": outputs}))
    usage.seconds = round(time.monotonic() - started, 1)
    for role, model in llms.items():
        m = model.get_token_usage_summary()
        # CrewAI's prompt_tokens include cache reads and writes; Anthropic's input_tokens do not
        uncached = m.prompt_tokens - m.cached_prompt_tokens - m.cache_creation_tokens
        cost = price(model_id, uncached, m.completion_tokens, m.cached_prompt_tokens, m.cache_creation_tokens)
        usage.roles[role] = {"calls": m.successful_requests, "cost_usd": round(cost or 0.0, 6)}
        usage.calls += m.successful_requests
        usage.input_tokens += uncached
        usage.output_tokens += m.completion_tokens
        usage.cache_read_input_tokens += m.cached_prompt_tokens
        usage.cache_creation_input_tokens += m.cache_creation_tokens
    usage.cost_usd = sum(r["cost_usd"] for r in usage.roles.values())
    usage.cost_source = "braid's prices on CrewAI's per-LLM token totals (summed over calls, so approximate)"
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description="F13: CrewAI (tester, implementer; two sequential tasks) on one exotic kata")
    parser.add_argument("--kata", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    report(run(args.kata, args.out.resolve(), args.model))


if __name__ == "__main__":
    main()
