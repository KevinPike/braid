"""The questions the harness asks tev1."""

from __future__ import annotations

from collections.abc import Mapping

from harness.decide.tev import Answer, Choice, Noul, Question, TevClient

ROUTER: Mapping[str, Question] = {
    "route": Choice(
        "Does the user's request need the assistant to look at files, folders, notes or the clock, "
        "or can it be answered by chatting?",
        {
            "chat": "General knowledge, writing, math, explanations, opinions or small talk. Nothing on the computer is needed.",
            "tool": "Needs reading a file, listing a folder, searching the user's notes or telling the current time or date.",
        },
    ),
}

COMPACTION: Mapping[str, Question] = {
    "compact": Noul(
        "The state describes how full the assistant's context window is. Should the conversation history be "
        "summarized now, before the next turn, to avoid running out of room?",
        {"true": "The window is nearly full or the next turn could overflow it.",
         "false": "There is plenty of room for several more turns."},
    ),
}


async def route(tev: TevClient, prompt: str) -> tuple[str, float]:
    """(``chat`` or ``tool``, its probability)."""
    answer: Answer = (await tev.ask("router", f"User request: {prompt}", ROUTER))["route"]
    assert answer.choice is not None
    return answer.choice, answer.probabilities.get(answer.choice, 0.0)


def compaction_state(*, used: int, num_ctx: int, growth: int, turns: int) -> str:
    return (
        f"Context window: {used} of {num_ctx} tokens used ({used / num_ctx:.0%}). "
        f"The largest turn so far added {growth} tokens. {turns} turns so far. "
        f"Room left: {num_ctx - used} tokens, about {max(0, (num_ctx - used) // max(growth, 1))} more turns of that size."
    )


async def should_compact(tev: TevClient, *, used: int, num_ctx: int, growth: int, turns: int) -> float:
    """tev1's probability that history should be compacted now."""
    answers = await tev.ask("compaction", compaction_state(used=used, num_ctx=num_ctx, growth=growth, turns=turns), COMPACTION)
    return answers["compact"].noul or 0.0
