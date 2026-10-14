"""Approval gate: no shell command or file write runs until the user approves it (ADR 0003).

A Strands hook on ``BeforeToolCallEvent``. For a gated tool it asks the Decision layer's policy check
for advice (optional, never trusted), then always awaits the Approver. There is no path on which the
advice alone lets the call proceed, and every failure (advisor, approver, a listener) cancels the tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from harness.hooks import ApprovalRequested, ApprovalResolved, HarnessHooks

# Tools that are gated whatever the configuration says. The shell's name is Strands' make_shell default.
ALWAYS_GATED: frozenset[str] = frozenset({"shell", "write_file", "run_skill_script"})
APPLY_WORKSPACE = "apply_workspace"  # the Approval a Workflow run asks before applying its diff (ADR 0005)
ADVICE_TIMEOUT_S = 10.0
PREVIEW_CHARS = 600


@dataclass(frozen=True)
class Advice:
    """A Decider's policy check (tev1 or Jev): probability the action is safe, or why there is none. Advisory only."""

    p_safe: float | None = None
    error: str = ""
    source: str = "tev1"

    def line(self) -> str:
        if self.p_safe is None:
            return f"{self.source}: unavailable ({self.error})" if self.error else f"{self.source}: unavailable"
        verdict = "safe" if self.p_safe >= 0.5 else "risky"
        return f"{self.source}: {verdict} p={self.p_safe:.2f} (advisory)"


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    input: Mapping[str, Any]
    advice: Advice = Advice()
    context: str = ""  # provenance the app adds for the prompt, for example a skill's declared allowed-tools

    def label(self) -> str:
        """One short line for the chat note: the shell command or the file path."""
        if self.tool == APPLY_WORKSPACE:
            n = len(self.input.get("changes", []))
            return f"apply the Workspace diff ({n} file{'' if n == 1 else 's'})"
        if self.tool == "run_skill_script":
            return f"run_skill_script: {self.input.get('name', '')}/{self.input.get('script', '')}"[:100]
        if self.tool.startswith("mcp__"):
            return f"{self.tool}: {json.dumps(self.input, sort_keys=True, default=str)[:80]}"
        key = {"shell": "command", "write_file": "path"}.get(self.tool)
        value = self.input.get(key) if key else None
        if value is None:
            value = json.dumps(self.input, sort_keys=True, default=str)
        text = str(value).replace("\n", " ")
        return f"{self.tool}: {text[:80]}{'…' if len(text) > 80 else ''}"

    def describe(self) -> str:
        """What the approval prompt shows: the exact arguments, with a clipped preview of file content."""
        if self.tool == APPLY_WORKSPACE:
            lines = [f"workspace: {self.input.get('workspace', '')}"]
            for c in self.input.get("changes", []):
                flags = f"  [{', '.join(c['flags'])}]" if c["flags"] else ""
                lines.append(f"  {c['status']:<8} {c['path']}{flags}")
            return "\n".join([*lines, str(self.input.get("diff", ""))])
        if self.tool == "shell":
            return f"command: {self.input.get('command', '')}"
        if self.tool == "run_skill_script":
            args = " ".join(str(a) for a in self.input.get("args") or ())
            lines = [
                f"skill: {self.input.get('name', '')}   script: {self.input.get('script', '')}   args: {args or '(none)'}",
                "runs in the Sandbox: the project writable, the skill's folder readable, no stdin, no API keys",
                "network: ALLOWED (a Sandbox escape)" if self.input.get("network") else "network: off",
            ]
            return "\n".join([*lines, self.context] if self.context else lines)
        if self.tool.startswith("mcp__"):  # an MCP server's tool (F16): the exact arguments, then where the server stands
            lines = ["arguments:", json.dumps(self.input, sort_keys=True, default=str, indent=2)]
            return "\n".join([*lines, self.context] if self.context else lines)
        if self.tool == "write_file":
            content = str(self.input.get("content", ""))
            preview = content[:PREVIEW_CHARS] + (f"\n[… {len(content) - PREVIEW_CHARS} more characters]" if len(content) > PREVIEW_CHARS else "")
            return f"path: {self.input.get('path', '')}\ncontent ({len(content)} characters):\n{preview}"
        return json.dumps(self.input, sort_keys=True, default=str, indent=2)


Approver = Callable[[ApprovalRequest], Awaitable[bool]]
# (tool name, input) -> extra lines for the prompt; shown beside the arguments, never read as advice
Annotator = Callable[[str, Mapping[str, Any]], str]
# (tool name, input) -> probability the action is safe, or None when there is no opinion
Advisor = Callable[[str, Mapping[str, Any]], Awaitable[float | None]]


class ApprovalGate(HookProvider):
    def __init__(
        self,
        approver: Approver,
        advisor: Advisor | None = None,
        *,
        gated: Collection[str] = ALWAYS_GATED,
        hooks: HarnessHooks | None = None,
        advice_timeout: float = ADVICE_TIMEOUT_S,
        annotate: Annotator | None = None,
        requires_approval: Callable[[str], bool] | None = None,
        advisor_name: Callable[[], str] = lambda: "tev1",
    ) -> None:
        self.advisor_name = advisor_name
        self.approver = approver
        self.advisor = advisor
        self.gated = frozenset(gated) | ALWAYS_GATED  # configuration can add tools, never drop the two
        self.hooks = hooks
        self.advice_timeout = advice_timeout
        self.annotate = annotate
        self.requires_approval = requires_approval  # a rule for names the set cannot list, such as mcp__* (F16)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before_tool)

    async def advise(self, tool: str, tool_input: Mapping[str, Any]) -> Advice:
        if self.advisor is None:
            return Advice()
        try:
            p = await asyncio.wait_for(self.advisor(tool, tool_input), self.advice_timeout)
        except Exception as exc:  # advice is optional: any failure just means "unavailable"
            return Advice(error=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__, source=self.advisor_name())
        return Advice(p_safe=p, source=self.advisor_name())

    def _rule(self, *names: str) -> bool:
        return self.requires_approval is not None and any(self.requires_approval(n) for n in names)

    async def before_tool(self, event: BeforeToolCallEvent) -> None:
        name = str(event.tool_use.get("name", ""))
        selected = event.selected_tool.tool_name if event.selected_tool is not None else name
        if name not in self.gated and selected not in self.gated and not self._rule(name, selected):
            return
        event.cancel_tool = f"{name} was not run: approval did not complete"  # stands unless the user approves
        raw = event.tool_use.get("input")
        request = ApprovalRequest(name, dict(raw) if isinstance(raw, Mapping) else {})
        try:
            context = self.annotate(name, request.input) if self.annotate is not None else ""
            request = ApprovalRequest(name, request.input, await self.advise(name, request.input), context)
            self._publish(ApprovalRequested(request))
            approved = await self.approver(request)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):  # a listener must not turn a cancel into something else
                self._publish(ApprovalResolved(request, False))
            raise
        except Exception as exc:
            event.cancel_tool = f"{name} was not run: approval failed ({type(exc).__name__}: {exc})"
            return
        approved = approved is True
        try:
            self._publish(ApprovalResolved(request, approved))
        except Exception as exc:
            event.cancel_tool = f"{name} was not run: approval failed ({type(exc).__name__}: {exc})"
            return
        event.cancel_tool = False if approved else f"{name} was denied by the user"

    def _publish(self, event: ApprovalRequested | ApprovalResolved) -> None:
        if self.hooks is not None:
            self.hooks.publish(event)
