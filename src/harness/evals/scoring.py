"""The Strands Evals side: the M4 prompts as ``Case``s, a deterministic evaluator, an ``Experiment`` per model.

Needs the optional ``evals`` dependency group (``uv run --group evals harness-eval``); ``runner`` imports this
module only when a model is actually run, so everything else works without it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationOutput

from harness.evals.report import CallLog, CaseRun


class EvalAgent(Protocol):
    """The slice of a Strands ``Agent`` the eval reads."""

    messages: list[Any]

    async def invoke_async(self, prompt: str) -> Any: ...


AgentFactory = Callable[[], EvalAgent]  # a fresh agent (empty history) per case


class FirstToolMatches(Evaluator[str, str]):
    """Pass when the first tool the agent called is the labelled one; with no label, when it called none.

    Deterministic on purpose: no LLM judge, so the score does not depend on a second model.
    """

    def evaluate(self, evaluation_case: Any) -> list[Any]:
        actual = evaluation_case.actual_trajectory
        if actual is None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason="the run failed before any trajectory was produced")]
        expected = evaluation_case.expected_trajectory or []
        first = actual[0] if actual else None
        want = expected[0] if expected else None
        ok = first == want
        return [EvaluationOutput(score=1.0 if ok else 0.0, test_pass=ok, reason=f"first tool {first or 'none'}, expected {want or 'none'}")]


def tools_used(messages: list[Any]) -> list[str]:
    return [b["toolUse"]["name"] for m in messages for b in m.get("content", []) if "toolUse" in b]


async def score_model(
    prompts: list[dict[str, Any]], factory: AgentFactory, log: CallLog
) -> list[CaseRun]:
    """Run every prompt on a fresh agent through a Strands Evals ``Experiment`` and return one ``CaseRun`` each."""
    runs: dict[str, CaseRun] = {}
    cases = [
        Case[str, str](
            name=f"{i:02d}", input=p["prompt"], expected_trajectory=[p["tool"]] if p.get("tool") else [],
        )
        for i, p in enumerate(prompts)
    ]

    async def task(case: Any) -> dict[str, Any]:
        start, seen = time.monotonic(), len(log.calls)
        expected = case.expected_trajectory[0] if case.expected_trajectory else None
        try:
            agent = factory()
            result = await agent.invoke_async(case.input)
            used = tools_used(agent.messages)
            err, output = "", str(result)
        except Exception as exc:  # a model that errors scores 0 on this case; the run goes on
            used, err, output = None, f"{type(exc).__name__}: {exc}", ""
        runs[case.name] = CaseRun(
            case.name, case.input, expected, used[0] if used else None, False,
            time.monotonic() - start, tuple(log.calls[seen:]), err,
        )
        return {"output": output, "trajectory": used}

    experiment: Any = Experiment[str, str](cases=cases, evaluators=[FirstToolMatches()])
    report = await experiment.run_evaluations_async(task, max_workers=1)  # one model, one GPU: sequential
    out: list[CaseRun] = []
    for row, passed in zip(report.cases, report.test_passes, strict=True):
        run = runs[row["name"]]
        out.append(CaseRun(run.name, run.prompt, run.expected, run.first_tool, bool(passed) and not run.error,
                           run.wall_s, run.calls, run.error))
    return out


__all__ = ["AgentFactory", "EvalAgent", "FirstToolMatches", "score_model"]
