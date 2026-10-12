"""F13 runner 3: the Claude Agent SDK, with a tester subagent and an implementer subagent.

    uv run --env-file .env --with claude-agent-sdk python examples/frameworks/run_agent_sdk.py --kata ferry-fare --out runs/agent-sdk
    uv run python examples/frameworks/grade.py --kata ferry-fare --out runs/agent-sdk

``claude-agent-sdk`` (checked against 0.2.165) is never a project dependency: ``uv run --with`` adds it for one run.
It drives the Claude Code CLI it bundles, which runs its own tool loop and builds its own requests. A coordinator
session delegates to two subagents (``AgentDefinition``s) through the ``Agent`` tool; routing is the coordinator
model's choice, as in a Swarm. The subagents run on ``--model`` too.

Restrictions (the CLI's Bash tool is not offered at all):

- ``tools``: Read, Write, Edit and Glob, plus the Agent tool, and one in-process MCP tool, ``mcp__kata__pytest``, which
  runs pytest in the kata copy with no credential in its environment (``common.run_tests``).
- A ``PreToolUse`` hook refuses any file path outside the kata copy or in a dot-directory, any write by the
  coordinator, a test-file write by the implementer, and a non-test write by the tester (the hook input's
  ``agent_type`` says which subagent is calling).
- ``setting_sources=[]``: no user or project settings, CLAUDE.md, hooks or MCP servers of the machine's own;
  ``strict_mcp_config``; ``permission_mode="dontAsk"`` so anything not pre-approved is denied.
- ``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1``: the CLI strips Anthropic and cloud credentials from any subprocess it starts.
  The CLI itself holds the key (it makes the API calls).

Isolation: model-written code runs here, outside braid's Sandbox and ADR 0005, in a temporary copy of the kata.

Usage: the CLI reports ``total_cost_usd`` and per-model usage in its ``ResultMessage``; calls are the distinct
assistant message ids, attributed to a Role through each message's ``parent_tool_use_id``. Note that if no
``ANTHROPIC_API_KEY`` is set, the CLI falls back to the machine's Claude login, and the cost is then notional. The trace
is ``<out>/.trace/messages.jsonl``, and the CLI's own transcript is under ``~/.claude/projects/`` (its session id is in
the JSON line).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

from common import (DEFAULT_MODEL, TASK, ToolRefused, Usage, export, implementer_prompt, is_test_path, kata_copy,
                    price, report, resolve, run_tests, tester_prompt, write_trace)

try:
    from claude_agent_sdk import (AgentDefinition, AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, HookMatcher,
                                  ResultMessage, ToolUseBlock, create_sdk_mcp_server, tool)
except ImportError as exc:  # pragma: no cover - only when run without --with claude-agent-sdk
    raise SystemExit("claude-agent-sdk is not installed: run this with `uv run --with claude-agent-sdk ...`") from exc

FILE_TOOLS = ["Read", "Write", "Edit", "Glob"]
PYTEST_TOOL = "mcp__kata__pytest"
WRITES = {"Write", "Edit"}
MAX_TURNS = 60
MAX_BUDGET_USD = 0.50

COORDINATOR = """You coordinate a two-role test-driven team; you never read or write files yourself.
1. Use the Agent tool to run the `tester` subagent: it writes pytest tests from GOAL.md.
2. Then run the `implementer` subagent: it writes the module until the tests pass.
3. If the implementer reports that a test contradicts GOAL.md, run the tester again with that report, then the
   implementer again. Do this at most twice.
Finish with one line: whether every test passes."""


def confine(work: Path) -> Any:
    """The PreToolUse hook: file paths stay in the kata copy, and each Role writes only its own files."""

    async def hook(data: Any, _tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        name = data.get("tool_name", "")
        args = data.get("tool_input", {}) or {}
        role = data.get("agent_type")  # absent on the coordinator's own thread
        try:
            if name in ("Read", *WRITES):
                rel = resolve(work, str(args.get("file_path", ""))).relative_to(work).as_posix()
                if name in WRITES:
                    if role not in ("tester", "implementer"):
                        raise ToolRefused("the coordinator does not write files")
                    if role == "tester" and not is_test_path(rel):
                        raise ToolRefused(f"the tester may write only test files, not {rel!r}")
                    if role == "implementer" and is_test_path(rel):
                        raise ToolRefused(f"the implementer may not write test files ({rel!r})")
            elif name == "Glob":
                resolve(work, str(args.get("path") or "."))
                pattern = str(args.get("pattern", ""))
                if pattern.startswith(("/", "~")) or ".." in pattern:
                    raise ToolRefused(f"glob pattern {pattern!r} leaves the kata directory")
        except ToolRefused as exc:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": str(exc)}}
        return {}

    return hook


def pytest_server(work: Path) -> Any:
    @tool("pytest", "Run pytest in the kata directory, optionally on one test file; returns the exit code and output.",
          {"path": str})
    async def pytest(args: dict[str, Any]) -> dict[str, Any]:
        try:
            output = await asyncio.to_thread(run_tests, work, str(args.get("path") or ""))
        except ToolRefused as exc:
            output = f"refused: {exc}"
        return {"content": [{"type": "text", "text": output}]}

    return create_sdk_mcp_server("kata", tools=[pytest])


def options(work: Path, kata: str, model_id: str) -> ClaudeAgentOptions:
    role_tools = [*FILE_TOOLS, PYTEST_TOOL]
    return ClaudeAgentOptions(
        model=model_id, cwd=str(work), system_prompt=COORDINATOR,
        tools=[*FILE_TOOLS, "Agent"], allowed_tools=[*FILE_TOOLS, "Agent", PYTEST_TOOL],
        mcp_servers={"kata": pytest_server(work)}, strict_mcp_config=True, setting_sources=[],
        permission_mode="dontAsk", max_turns=MAX_TURNS, max_budget_usd=MAX_BUDGET_USD,
        agents={
            "tester": AgentDefinition(description="Writes pytest tests from GOAL.md; never the implementation.",
                                      prompt=tester_prompt(kata), tools=role_tools, model=model_id),
            "implementer": AgentDefinition(description="Writes the module so the tester's tests pass; never edits tests.",
                                           prompt=implementer_prompt(kata), tools=role_tools, model=model_id),
        },
        hooks={"PreToolUse": [HookMatcher(matcher="Read|Write|Edit|Glob", hooks=[confine(work)])]},
        env={"CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1"},
        forward_subagent_text=True,  # the subagents' text too, so every call they make shows in the stream and the trace
    )


def record(message: Any) -> dict[str, Any]:
    data = dataclasses.asdict(message) if dataclasses.is_dataclass(message) and not isinstance(message, type) else {"repr": repr(message)}
    return {"type": type(message).__name__, **data}


async def run(kata: str, out: Path, model_id: str) -> Usage:
    usage = Usage("claude-agent-sdk", kata, model_id, str(out))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        usage.notes.append("no ANTHROPIC_API_KEY: the CLI used the machine's Claude login, so cost_usd is notional")
    started = time.monotonic()
    spawned: dict[str, str] = {}  # Agent tool_use id -> subagent name
    seen: dict[str, tuple[str, dict[str, Any]]] = {}  # message id -> (role, usage): one entry per API call
    log: list[dict[str, Any]] = []
    result: ResultMessage | None = None
    with kata_copy(kata) as work:
        try:
            async with ClaudeSDKClient(options(work, kata, model_id)) as client:
                await client.query(TASK)
                async for message in client.receive_response():
                    log.append(record(message))
                    if isinstance(message, AssistantMessage):
                        role = spawned.get(message.parent_tool_use_id or "", "coordinator")
                        for block in message.content:
                            if isinstance(block, ToolUseBlock) and block.name in ("Agent", "Task"):
                                spawned[block.id] = str(block.input.get("subagent_type", "?"))
                        if message.message_id and message.usage:
                            seen[message.message_id] = (role, message.usage)
                    elif isinstance(message, ResultMessage):
                        result = message
        except Exception as exc:  # a failed run still reports what it spent, and keeps what it wrote
            usage.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        export(work, out)
        usage.trace = str(write_trace(out, "messages.jsonl", "\n".join(json.dumps(x, default=str) for x in log) + "\n"))
    usage.seconds = round(time.monotonic() - started, 1)
    for role, u in seen.values():
        tokens = (int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                  int(u.get("cache_read_input_tokens", 0)), int(u.get("cache_creation_input_tokens", 0)))
        tally = usage.roles.setdefault(role, {"calls": 0, "cost_usd": 0.0})
        tally["calls"] += 1
        tally["cost_usd"] = round(tally["cost_usd"] + (price(model_id, *tokens) or 0.0), 6)
        usage.calls += 1
        usage.input_tokens += tokens[0]
        usage.output_tokens += tokens[1]
        usage.cache_read_input_tokens += tokens[2]
        usage.cache_creation_input_tokens += tokens[3]
    if result is not None:
        usage.cost_usd = result.total_cost_usd
        usage.cost_source = "the CLI's ResultMessage.total_cost_usd (per-Role costs: braid's prices on each message's usage)"
        if usage.status == "ok":
            usage.status = result.subtype if not result.is_error else f"error: {result.subtype}"
        usage.notes.append(f"session {result.session_id}, {result.num_turns} turns")
    else:
        usage.cost_usd = sum(r["cost_usd"] for r in usage.roles.values())
        usage.cost_source = "braid's prices on each message's usage (no ResultMessage)"
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description="F13: Claude Agent SDK subagents (tester, implementer) on one exotic kata")
    parser.add_argument("--kata", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    report(asyncio.run(run(args.kata, args.out.resolve(), args.model)))


if __name__ == "__main__":
    main()
