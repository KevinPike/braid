"""The Skills plugin (F14, F15): the Skill index in the system prompt, ``skill`` and ``read_skill_file`` as tools.

Built on Strands' public ``Skill`` parser rather than its ``AgentSkills`` plugin, which emits absolute paths, reads
through the agent's sandbox and has no tiers (ADR 0011). Only names and descriptions reach the model up front; a body
loads when the model calls ``skill``, or the user types ``/skill-name``. Bodies are ordinary history: Compaction may drop
one, and the index then says so (D9).
"""

from __future__ import annotations

import re
from typing import Any

from strands import Agent
from strands.hooks.events import BeforeInvocationEvent
from strands.plugins import Plugin, hook
from strands.tools.decorator import tool
from strands.types.content import Message
from strands.types.tools import ToolContext

from harness.context.accounting import TokenEstimator
from harness.context.compaction import SUMMARY_HEADER
from harness.context.trimlog import TrimEntry, TrimLog
from harness.skills.catalog import SkillCatalog

SKILL_TOOLS = frozenset({"skill", "read_skill_file"})  # the offloader keeps these results in context (D9)
SKILL_BODY_TOKENS = 5000  # ...up to this many tokens; a bigger one is offloaded like any large result
STATE_KEY = "skills"
LOADED = re.compile(r'<skill name="([a-z0-9-]+)">')


def skill_text(catalog: SkillCatalog, name: str) -> str:
    """A skill's body as the model sees it: tagged with its name (so Compaction can tell it is still there), then the
    resources it can read or run, then what the skill declares for ``allowed-tools`` (shown, never honoured)."""
    entry = catalog.entry(name)
    parts = [f'<skill name="{name}">\n{catalog.body(name)}\n</skill>']
    resources = catalog.resources(name)
    if resources:
        parts.append("Files in this skill (read with read_skill_file; run scripts/ with run_skill_script):\n" + "\n".join(f"  {r}" for r in resources))
    if entry.allowed_tools:
        parts.append(f"The skill declares allowed-tools: {' '.join(entry.allowed_tools)} (display only: every gated tool still needs the user's approval).")
    return "\n\n".join(parts)


def invocation_turn(catalog: SkillCatalog, name: str, args: str) -> str:
    """The user turn a ``/skill-name args`` becomes: the body plus the arguments."""
    lead = f"The user invoked the skill {name!r}. Follow its instructions for this request."
    return f"{lead}\n\n{skill_text(catalog, name)}\n\nArguments: {args or '(none)'}"


def loaded_names(messages: list[Message]) -> set[str]:
    """Skills whose body is still in ``messages`` (a Summary does not count: it holds only the summarizer's words)."""
    out: set[str] = set()
    for message in messages:
        for block in message["content"]:
            if "text" in block:
                texts = [block["text"]]
            elif "toolResult" in block:
                texts = [str(c.get("text", "")) for c in block["toolResult"]["content"]]
            else:
                continue
            out.update(n for t in texts if not t.startswith(SUMMARY_HEADER) for n in LOADED.findall(t))
    return out


class SkillsPlugin(Plugin):
    name = "braid_skills"

    def __init__(self, catalog: SkillCatalog, *, trimlog: TrimLog | None = None, estimator: TokenEstimator | None = None) -> None:
        self.catalog = catalog
        self._trimlog = trimlog
        self._estimator = estimator or TokenEstimator()
        super().__init__()

    # --- state: agent.state["skills"] = {"active": [...], "dropped": [...], "injected": "..."} -----------------------------

    @staticmethod
    def _state(agent: Agent) -> dict[str, Any]:
        state = agent.state.get(STATE_KEY)
        return state if isinstance(state, dict) else {"active": [], "dropped": [], "injected": ""}

    def activate(self, agent: Agent, name: str) -> None:
        state = self._state(agent)
        if name not in state["active"]:
            state["active"] = [*state["active"], name]
        agent.state.set(STATE_KEY, state)

    # --- tools -------------------------------------------------------------------------------------------------------

    @tool(context=True)
    def skill(self, name: str, tool_context: ToolContext) -> str:
        """Load a skill's instructions. Use it when a task matches a skill listed in available_skills.

        Args:
            name: The skill's name, as listed.
            tool_context: Injected by the framework. Not user-facing.
        """
        try:
            text = skill_text(self.catalog, name)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        self.activate(tool_context.agent, name)
        return text

    @tool
    def read_skill_file(self, name: str, path: str) -> str:
        """Read a file that belongs to a skill (a reference or an asset the instructions name).

        Args:
            name: The skill's name.
            path: The file's path inside the skill, as the skill lists it, for example references/format.md.
        """
        try:
            return self.catalog.resource(name, path)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None

    # --- the index ---------------------------------------------------------------------------------------------------

    @hook
    def _before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Write the index block into the system prompt. It is the same bytes every turn until the skill set changes or
        Compaction drops an active body, so the cached prefix stays valid."""
        agent = event.agent
        self.catalog.reload()
        state = self._state(agent)
        present = loaded_names([*agent.messages, *(event.messages or [])])
        dropped = [n for n in state["active"] if n not in present and self.catalog.has(n)]
        for name in dropped:
            if name not in state["dropped"] and self._trimlog is not None:
                before = self._estimator.tokens(self.catalog.entry(name).body_chars)
                self._trimlog.record(TrimEntry(
                    "skill", f"Compaction dropped the body of the active skill {name}", 0, before, 0,
                    f"the index now says: active: {name} (reload with skill)",
                ))
        block = self.catalog.index_block()
        if block and dropped:
            block += "\n" + "\n".join(f"active: {n} (reload with `skill`)" for n in dropped)
        prompt = agent.system_prompt or ""
        if state["injected"] and state["injected"] in prompt:
            prompt = prompt.replace(state["injected"], "")
        injection = f"\n\n{block}" if block and prompt else block
        agent.system_prompt = prompt + injection
        agent.state.set(STATE_KEY, {"active": state["active"], "dropped": dropped, "injected": injection})
