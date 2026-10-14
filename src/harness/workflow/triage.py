"""Error triage (ADR 0014): a Decider sorts a failed test run into one of a fixed set of kinds, and the kind selects a
canned note for the next implementation prompt.

The Decider never routes. Code still decides every edge (ADR 0004, 0008); the note is text the implementer reads, chosen
from a list written here, so a wrong or missing answer costs one sentence of guidance and nothing else. Below
``min_p``, or on any failure, there is no note. Its cost counts toward the run's Spend cap.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from harness.config import HarnessConfig
from harness.decide.jev import JevDecider
from harness.decide.tev import Choice, Decider

MAX_OUTPUT_CHARS = 6000  # the tail of the run: that is where pytest puts the failures and the summary

# kind -> (what it looks like, the note the implementer reads)
KINDS: Mapping[str, tuple[str, str]] = {
    "setup_error": (
        "pytest could not import, collect or start: a SyntaxError, ImportError, NameError at import time, or a missing "
        "module or file. No test ran.",
        "The last run did not get as far as running the tests (an import, syntax or name error). Fix that error first; "
        "do not change behavior until the tests run.",
    ),
    "wrong_value": (
        "Tests ran and an assert failed because the value the code returned differs from the one expected.",
        "The tests ran and a value was wrong. Compare the expected and actual values in the output, find the rule that "
        "produces the difference, and fix the computation. Do not special-case the failing input.",
    ),
    "exception": (
        "Tests ran and the code raised an exception such as TypeError, KeyError, IndexError, ZeroDivisionError or "
        "AttributeError.",
        "The code raised an exception. Read the last frame of the traceback that is in your implementation file and fix "
        "the input or case it does not handle.",
    ),
    "hang": (
        "The run timed out, or a loop or recursion did not end or was far too slow.",
        "The last run timed out. Look for a loop or recursion that does not end for some input, or one that repeats "
        "work, and fix that.",
    ),
    "regression": (
        "A test from an earlier behavior fails, not only the new one.",
        "An earlier behavior's test fails now. Keep the earlier behavior working while you add the new one; do not "
        "rewrite what already passed.",
    ),
}

QUESTIONS: Mapping[str, Choice] = {
    "kind": Choice(
        "The state is the output of a failed pytest run on code under test-driven development. What kind of failure is it?",
        {name: criterion for name, (criterion, _) in KINDS.items()},
    ),
}


@dataclass(frozen=True)
class Triaged:
    kind: str
    p: float
    note: str  # "" when the answer was below the threshold
    cost_usd: float


class Triage:
    def __init__(self, decider: Decider, *, min_p: float = 0.6) -> None:
        self.decider = decider
        self.min_p = min_p

    async def classify(self, output: str) -> Triaged | None:
        """The kind of failure ``output`` shows, or None when the Decider failed (the failure is in the Decision log)."""
        before = float(getattr(self.decider, "cost_usd", 0.0))
        try:
            answer = (await self.decider.ask("triage", output[-MAX_OUTPUT_CHARS:], QUESTIONS))["kind"]
        except Exception:
            return None
        cost = float(getattr(self.decider, "cost_usd", 0.0)) - before
        kind = answer.choice
        if kind is None or kind not in KINDS:
            return None
        p = answer.probabilities.get(kind, 0.0)
        return Triaged(kind, p, KINDS[kind][1] if p >= self.min_p else "", cost)


def make_triage(config: HarnessConfig, *, environ: Mapping[str, str] | None = None) -> Triage | None:
    """The run's Triage, or None when ``[workflow] triage`` is off or there is no OpenRouter key (a run never needs it)."""
    if not config.workflow.triage:
        return None
    key = (environ if environ is not None else os.environ).get("OPENROUTER_API_KEY", "")
    if not key:
        return None
    dec = config.decide
    jev = JevDecider(key, dec.openrouter_model, timeout_s=dec.timeout_s, max_cost_usd=config.workflow.max_cost_usd)
    return Triage(jev, min_p=config.workflow.triage_min_p)
