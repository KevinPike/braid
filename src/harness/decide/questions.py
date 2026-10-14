"""The questions the harness asks tev1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harness.decide.tev import Decider, Noul, Question

COMPACTION: Mapping[str, Question] = {
    "compact": Noul(
        "The state describes how full the assistant's context window is. Should the conversation history be "
        "summarized now, before the next turn, to avoid running out of room?",
        {"true": "The window is nearly full or the next turn could overflow it.",
         "false": "There is plenty of room for several more turns."},
    ),
}

POLICY: Mapping[str, Question] = {
    "safe": Noul(
        "The state describes one action the assistant wants to run in the user's project folder. Is it safe and "
        "low-risk: it only reads or adds files inside the folder and does not delete or overwrite important data, "
        "reach outside the folder, use the network, install software or need elevated privileges?",
        {"true": "A harmless, easily undone action confined to the project folder.",
         "false": "It deletes, overwrites, escapes the folder, uses the network, installs software or is unclear."},
    ),
}
POLICY_VALUE_CHARS = 500


def compaction_state(*, used: int, num_ctx: int, growth: int, turns: int) -> str:
    return (
        f"Context window: {used} of {num_ctx} tokens used ({used / num_ctx:.0%}). "
        f"The largest turn so far added {growth} tokens. {turns} turns so far. "
        f"Room left: {num_ctx - used} tokens, about {max(0, (num_ctx - used) // max(growth, 1))} more turns of that size."
    )


async def should_compact(tev: Decider, *, used: int, num_ctx: int, growth: int, turns: int) -> float:
    """tev1's probability that history should be compacted now."""
    answers = await tev.ask("compaction", compaction_state(used=used, num_ctx=num_ctx, growth=growth, turns=turns), COMPACTION)
    return answers["compact"].noul or 0.0


def policy_state(tool: str, args: Mapping[str, Any]) -> str:
    """The action as a short text: tool name plus each argument clipped, so a large file body never reaches tev1."""
    lines = [f"Action: {tool}"]
    for key, value in args.items():
        text = str(value)
        if len(text) > POLICY_VALUE_CHARS:
            text = f"{text[:POLICY_VALUE_CHARS]}[... {len(text) - POLICY_VALUE_CHARS} more characters]"
        lines.append(f"{key}: {text}")
    return "\n".join(lines)


async def check_policy(tev: Decider, tool: str, args: Mapping[str, Any]) -> float:
    """tev1's probability that the action is safe to run. Advice for the approval prompt, never a decision."""
    answers = await tev.ask("policy", policy_state(tool, args), POLICY)
    return answers["safe"].noul or 0.0
