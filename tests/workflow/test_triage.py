"""ADR 0014: error triage. A Decider sorts a failed Green run into a kind, the kind picks a canned note for the next
implementation prompt, and code routes exactly as before."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.config import HarnessConfig
from harness.decide.tev import Answer, Question
from harness.workflow.triage import KINDS, Triage, make_triage
from tests.workflow.test_execute import (
    MUL_IMPL, MUL_TEST, NEG, NEG_TEST, NEG_WRONG, ScriptedModel, edit, run_loop,
)


@dataclass
class FakeDecider:
    """Answers ``kind`` with a fixed choice and probability, and spends ``cost`` a call, like JevDecider.cost_usd."""

    choice: str = "wrong_value"
    p: float = 0.9
    cost: float = 0.00002
    fail: bool = False
    model: str = "fake-jev"
    cost_usd: float = 0.0
    asked: list[str] = field(default_factory=list)

    async def ask(self, kind: str, state: str | Mapping[str, Any] | list[Any], questions: Mapping[str, Question]) -> dict[str, Answer]:
        self.asked.append(str(state))
        if self.fail:
            raise TimeoutError("decider down")
        self.cost_usd += self.cost
        return {"kind": Answer("choice", choice=self.choice, probabilities={self.choice: self.p})}


def script() -> ScriptedModel:
    polished = edit("src/mod.py", '"""Helpers."""\n\n' + NEG)
    return ScriptedModel([MUL_TEST, MUL_IMPL, NEG_TEST, edit("src/mod.py", NEG_WRONG), edit("src/mod.py", NEG), polished])


NOTE = KINDS["wrong_value"][1]


async def test_a_failed_run_adds_the_canned_note_and_the_cost_counts_but_the_route_is_unchanged(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    decider, model = FakeDecider(), script()
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "yes"], triage=Triage(decider))
    assert run.rec.trace == [  # the same nodes as the happy path with no Decider: code routes
        "check_assumptions",
        "write_test", "run_red", "write_impl", "run_green", "next_behavior",
        "write_test", "run_red", "write_impl", "run_green", "write_impl", "run_green", "next_behavior",
        "sign_off", "polish", "verify_polish", "finish",
    ]
    impl_prompts = [p for p in model.prompts if "Task: make the tests pass" in p]
    assert NOTE not in impl_prompts[0] and NOTE not in impl_prompts[1] and NOTE in impl_prompts[2]  # only after the failure
    assert len(decider.asked) == 1 and "assert" in decider.asked[0]  # passing runs are not triaged
    assert run.rec.cost_usd == 0.00002 and run.rec.roles["decider"] == {"calls": 1, "cost_usd": 0.00002}
    green = [line for line in run.log.lines() if line["node"] == "run_green"]
    assert [g.get("triage") for g in green] == [None, {"kind": "wrong_value", "p": 0.9, "noted": True, "cost_usd": 0.00002}, None]


async def test_below_the_threshold_there_is_no_note(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = script()
    await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "yes"], triage=Triage(FakeDecider(p=0.4), min_p=0.6))
    assert not any(NOTE in p for p in model.prompts)


async def test_a_decider_that_fails_costs_the_run_nothing(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = script()
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "yes"], triage=Triage(FakeDecider(fail=True)))
    assert run.final.get("status") != "aborted" and run.rec.cost_usd == 0.0 and not any(NOTE in p for p in model.prompts)


async def test_an_unknown_kind_is_ignored() -> None:
    assert await Triage(FakeDecider(choice="mystery")).classify("boom") is None


def test_triage_is_off_by_default_and_without_a_key_the_run_goes_without() -> None:
    config = HarnessConfig()
    assert make_triage(config, environ={"OPENROUTER_API_KEY": "k"}) is None
    config.workflow.triage = True
    assert make_triage(config, environ={}) is None
    assert make_triage(config, environ={"OPENROUTER_API_KEY": "k"}) is not None
