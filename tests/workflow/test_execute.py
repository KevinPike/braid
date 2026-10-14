"""M13 graph gates: the Execution loop with a scripted model, LocalSandbox and a fixture repo with a venv."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from harness.config import WorkflowConfig
from harness.sandbox import LocalSandbox, Sandbox, Workspace
from harness.workflow.execute import Dispute, Edits, ExecState, FileEdit, ImplEdits, build_execution_graph, drive_graph, initial_state
from harness.workflow.cli import show
from harness.workflow.invariants import Invariant
from harness.workflow.model import WorkflowModel
from harness.workflow.plan import Assumption, Plan
from harness.workflow.triage import Triage
from harness.workflow.records import Recorder, RunLog

T = TypeVar("T", bound=BaseModel)

MOD = "def add(a: int, b: int) -> int:\n    return a + b\n"
MUL = MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a * b\n"
NEG_WRONG = MUL + "\n\ndef neg(a: int) -> int:\n    return a\n"
NEG = MUL + "\n\ndef neg(a: int) -> int:\n    return -a\n"


def edit(path: str, content: str) -> Edits:
    return Edits(edits=[FileEdit(path=path, content=content)], rationale="scripted")


def make_test(n: int, body: str, imports: str = "mul") -> Edits:
    return edit(f"tests/test_behavior_{n}.py", f"from src.mod import {imports}\n\n\ndef test_b{n}() -> None:\n    {body}\n")


MUL_TEST = make_test(1, "assert mul(2, 3) == 6")
MUL_IMPL = edit("src/mod.py", MUL)
NEG_TEST = make_test(2, "assert neg(3) == -3", "neg")


@dataclass
class ScriptedModel:
    """Replies in call order; a reply may be a callable of the prompt. Records every prompt."""

    replies: list[BaseModel | Callable[[str], BaseModel]]
    prompts: list[str] = field(default_factory=list)

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        assert self.replies, f"the scripted model ran out of replies; prompt was:\n{prompt}"
        reply = self.replies.pop(0)
        edits = reply(prompt) if callable(reply) else reply
        if isinstance(edits, schema):
            return edits
        return schema.model_validate(edits.model_dump())  # write_impl asks for ImplEdits: an Edits reply without a Dispute


@dataclass
class Run:
    repo: Path
    ws: Workspace
    rec: Recorder
    log: RunLog
    final: ExecState
    payloads: list[dict[str, Any]]


def tree(root: Path) -> dict[str, str]:
    skip = {".git", ".venv", ".tmp", "__pycache__"}
    return {
        str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*"))
        if p.is_file() and not skip & set(p.relative_to(root).parts)
    }


async def run_loop(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, model: WorkflowModel, answers: list[str], *,
    behaviors: list[str] | None = None, assumptions: list[Assumption] | None = None, hints: bool = True,
    settings: WorkflowConfig | None = None, invariants: list[Invariant] | None = None, arbiter: WorkflowModel | None = None,
    sandbox: Sandbox | None = None, tester: WorkflowModel | None = None, implementer: WorkflowModel | None = None,
    triage: Triage | None = None,
) -> Run:
    ws = await Workspace.create(fixture_repo, tmp_path / "workspaces", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    plan = Plan(
        goal="arithmetic helpers", behaviors=behaviors or ["mul multiplies", "neg negates"],
        assumptions=assumptions if assumptions is not None else [Assumption(text="pyproject exists", probe="test -f pyproject.toml")],
        files=["src/mod.py"], invariants=invariants or [],
    )
    rec = Recorder()
    log = RunLog(tmp_path / "runs" / "run1.jsonl")
    graph = build_execution_graph(
        model, sandbox or LocalSandbox(), ws, settings or WorkflowConfig(), hints=hints, recorder=rec, log=log, arbiter=arbiter,
        tester=tester, implementer=implementer, triage=triage,
    )
    payloads: list[dict[str, Any]] = []

    def ask(payload: dict[str, Any]) -> str:
        payloads.append(payload)
        assert answers, f"no scripted answer for {payload['kind']} interrupt"
        return answers.pop(0)

    final = await drive_graph(graph, initial_state(plan), ask)
    return Run(fixture_repo, ws, rec, log, final, payloads)


async def test_happy_path_two_behaviors_wrong_once_ends_at_finish_and_applies(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    polished = edit("src/mod.py", '"""Helpers."""\n\n' + NEG)
    model = ScriptedModel([MUL_TEST, MUL_IMPL, NEG_TEST, edit("src/mod.py", NEG_WRONG), edit("src/mod.py", NEG), polished])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "yes"])
    assert run.rec.trace == [
        "check_assumptions",
        "write_test", "run_red", "write_impl", "run_green", "next_behavior",
        "write_test", "run_red", "write_impl", "run_green", "write_impl", "run_green", "next_behavior",
        "sign_off", "polish", "verify_polish", "finish",
    ]
    assert [p["kind"] for p in run.payloads] == ["sign_off", "finish"]
    assert run.final["status"] == "applied" and run.rec.polish == "kept"
    assert [b.attempts for b in run.rec.behaviors] == [1, 2]
    files = tree(fixture_repo)
    assert files["src/mod.py"].startswith('"""Helpers."""') and "def neg" in files["src/mod.py"] and "def mul" in files["src/mod.py"]
    assert {"tests/test_behavior_1.py", "tests/test_behavior_2.py"} <= set(files)
    assert not run.ws.path.exists()  # applied, then removed


async def test_a_test_that_passes_at_red_and_at_head_routes_back_to_write_test_and_three_in_a_row_get_stuck(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    vacuous = make_test(1, "assert add(1, 2) == 3", "add")
    model = ScriptedModel([vacuous, vacuous, vacuous])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"])
    assert run.rec.trace == ["check_assumptions"] + ["write_test", "run_red", "check_satisfied"] * 3 + ["stuck"]
    assert run.payloads[0]["kind"] == "stuck"
    stats = run.rec.behaviors[0]
    assert stats.red_rejections == 3 and not stats.satisfied and not stats.green
    assert "passes even without any implementation" in model.prompts[1] + model.prompts[2]  # the rewrite is told why


async def test_a_test_that_passes_at_red_but_fails_at_head_marks_the_behavior_satisfied_and_leaves_the_workspace_alone(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    again = make_test(2, "assert mul(3, 3) == 9")  # mul exists since Behavior 1; it did not at HEAD
    model = ScriptedModel([MUL_TEST, MUL_IMPL, again])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"])
    assert run.rec.trace[6:11] == ["write_test", "run_red", "check_satisfied", "next_behavior", "sign_off"]
    first, second = run.rec.behaviors
    assert second.satisfied and second.green and second.attempts == 0 and not first.satisfied
    assert tree(run.ws.path)["src/mod.py"] == MUL  # the HEAD swap was undone
    assert "tests/test_behavior_2.py" in tree(run.ws.path) and not (run.ws.path / "src" / "__pycache__").exists()
    swap = next(line for line in run.log.lines() if line["node"] == "check_satisfied")
    assert swap["satisfied"] is True and swap["swapped"] == ["src/mod.py"]


async def test_a_syntax_error_in_the_test_routes_to_write_test_never_write_impl(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    broken = edit("tests/test_behavior_1.py", "def test_x(:\n    pass\n")
    model = ScriptedModel([broken, MUL_TEST, MUL_IMPL])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"], behaviors=["mul multiplies"])
    assert run.rec.trace[:6] == ["check_assumptions", "write_test", "run_red", "write_test", "run_red", "write_impl"]
    assert run.rec.behaviors[0].red_rejections == 1


async def test_green_failing_max_attempts_times_gets_stuck_and_a_hint_resumes_write_impl(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    wrong = edit("src/mod.py", MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a + b\n")
    model = ScriptedModel([MUL_TEST, wrong, wrong, wrong, MUL_IMPL])
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, ["multiply them", "abort"], behaviors=["mul multiplies"],
    )
    assert run.rec.trace[:12] == [
        "check_assumptions", "write_test", "run_red",
        "write_impl", "run_green", "write_impl", "run_green", "write_impl", "run_green", "stuck", "write_impl", "run_green",
    ]
    assert run.payloads[0]["kind"] == "stuck" and "assert" in run.payloads[0]["last_run"]
    assert "multiply them" in model.prompts[-1] and "multiply them" not in model.prompts[-2]
    assert run.rec.behaviors[0].hints == 1 and run.rec.behaviors[0].green


async def test_a_failing_probe_interrupts_at_clarify_before_any_file_is_written(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, ScriptedModel([]), ["abort"],
        assumptions=[Assumption(text="a file exists", probe="test -f nope.txt")],
    )
    assert run.rec.trace == ["check_assumptions", "clarify"]
    assert run.payloads[0] == {"kind": "clarify", "failed": ["a file exists"], "open_questions": []}
    assert await run.ws.changes() == [] and run.final["status"] == "aborted"


async def test_an_answer_to_clarify_waives_the_failed_assumption_and_the_run_goes_on(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, MUL_IMPL])
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, ["fine, ignore it", "abort"], behaviors=["mul multiplies"],
        assumptions=[Assumption(text="a file exists", probe="test -f nope.txt")],
    )
    assert run.rec.trace[:5] == ["check_assumptions", "clarify", "check_assumptions", "write_test", "run_red"]
    assert "fine, ignore it" in model.prompts[0]  # the answer became a note


async def test_a_sign_off_answer_other_than_yes_or_abort_adds_a_behavior_and_returns_to_write_test(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, MUL_IMPL, NEG_TEST, edit("src/mod.py", NEG)])
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, ["it must also negate", "abort"], behaviors=["mul multiplies"],
    )
    assert run.rec.trace[6:10] == ["sign_off", "write_test", "run_red", "write_impl"]
    assert run.final["plan"]["behaviors"] == ["mul multiplies", "it must also negate"]
    assert "it must also negate" in model.prompts[2]
    assert [p["kind"] for p in run.payloads] == ["sign_off", "sign_off"]


async def test_a_polish_edit_that_breaks_a_test_is_reverted(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = ScriptedModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", "def add(a, b):\n    return 0\n")])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"])
    assert run.rec.trace[-4:] == ["sign_off", "polish", "verify_polish", "finish"]
    assert run.rec.polish == "reverted"
    assert tree(run.ws.path)["src/mod.py"] == MUL  # equals its pre-Polish contents


async def test_a_file_edit_outside_the_workspace_is_not_written_and_counts_as_an_attempt(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    escape = edit("../outside.py", "X = 1\n")
    model = ScriptedModel([MUL_TEST, escape, MUL_IMPL])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"], behaviors=["mul multiplies"])
    assert not (run.ws.path.parent / "outside.py").exists() and not (tmp_path / "outside.py").exists()
    assert run.rec.behaviors[0].attempts == 2
    assert run.rec.trace[3:7] == ["write_impl", "run_green", "write_impl", "run_green"]
    refusal = next(line for line in run.log.lines() if line["node"] == "write_impl")
    assert refusal["written"] == [] and "outside" in refusal["refused"][0]
    assert "Edit refused" in model.prompts[-1] and run.rec.behaviors[0].green


async def test_answering_no_at_finish_leaves_the_project_unchanged_and_the_workspace_on_disk(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    before = tree(fixture_repo)
    model = ScriptedModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"])
    assert run.rec.trace[-1] == "finish" and run.final["status"] == "kept"
    finish = run.payloads[-1]
    assert finish["kind"] == "finish" and {c["path"] for c in finish["changes"]} == {"src/mod.py", "tests/test_behavior_1.py"}
    assert "def mul" in finish["diff"]
    assert tree(fixture_repo) == before
    assert run.ws.path.exists() and "def mul" in (run.ws.path / "src" / "mod.py").read_text()


async def test_no_hints_mode_skips_a_stuck_behavior_and_restores_the_workspace(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    wrong = edit("src/mod.py", MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a + b\n")
    model = ScriptedModel([MUL_TEST, wrong, wrong, wrong, edit("src/mod.py", MOD)])
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, ["no"], behaviors=["mul multiplies"], hints=False,
        settings=WorkflowConfig(max_polish_rounds=1),
    )
    assert run.rec.stuck == 1 and not run.rec.behaviors[0].green
    assert run.rec.trace[:10] == [
        "check_assumptions", "write_test", "run_red", "write_impl", "run_green", "write_impl", "run_green", "write_impl",
        "run_green", "stuck",
    ]
    assert tree(run.ws.path) == tree(fixture_repo)  # the failed Behavior's test and code are gone
    assert [p["kind"] for p in run.payloads] == ["finish"]  # finish still asks


async def test_the_run_log_has_one_rich_line_per_node(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = ScriptedModel([MUL_TEST, MUL_IMPL])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"], behaviors=["mul multiplies"])
    lines = [line for line in run.log.lines() if line["node"] != "route"]  # Route records are tested in test_trace
    assert [line["node"] for line in lines] == run.rec.trace
    impl = next(line for line in lines if line["node"] == "write_impl")
    assert "mul multiplies" in impl["prompt"] and impl["reply"]["edits"][0]["path"] == "src/mod.py"
    assert impl["written"] == ["src/mod.py"] and impl["attempt"] == 1 and impl["behavior"] == 0 and impl["model_seconds"] >= 0
    green = next(line for line in lines if line["node"] == "run_green")
    assert green["exit_code"] == 0 and "passed" in green["output_tail"] and green["timed_out"] is False
    red = next(line for line in lines if line["node"] == "run_red")
    assert red["exit_code"] != 0 and red["verdict"] == "red"
    assert all("ts" in line for line in lines)


@pytest.mark.parametrize("answer", ["y", "yes", "OK"])
async def test_yes_variants_approve_sign_off(fixture_repo: Path, fixture_venv: Path, tmp_path: Path, answer: str) -> None:
    model = ScriptedModel([MUL_TEST, MUL_IMPL, MUL_IMPL])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, [answer, "no"], behaviors=["mul multiplies"])
    assert run.rec.trace[-3:] == ["polish", "verify_polish", "finish"]


# --- F3: Invariants and Disputes ---------------------------------------------------------------------------------------

MUL_RULE = Invariant(text="mul(a, b) is a times b", fn="mul", check="result == args[0] * args[1]")
BAD_MUL_TEST = make_test(1, "assert mul(2, 3) == 7")


def dispute(test: str, reason: str, *edits: FileEdit) -> ImplEdits:
    return ImplEdits(edits=list(edits), rationale="scripted", dispute=Dispute(test=test, reason=reason))


def nodes(run: Run, name: str) -> list[dict[str, Any]]:
    return [r for r in run.log.lines() if r["node"] == name]


async def test_a_new_test_that_breaks_an_invariant_goes_back_to_write_test_before_implementation_sees_it(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([BAD_MUL_TEST, MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], invariants=[MUL_RULE])
    assert run.rec.trace[:7] == ["check_assumptions", "write_test", "run_red", "check_invariants", "write_test", "run_red", "check_invariants"]
    first, second = nodes(run, "check_invariants")
    assert first["violations"][0]["invariant"] == MUL_RULE.text and "mul(2, 3) == 7" in first["violations"][0]["example"]
    assert second["violations"] == [] and second["checked"] == 1
    rewrite = model.prompts[1]
    assert MUL_RULE.text in rewrite and "mul(2, 3) == 7" in rewrite  # the rule and the example are named
    impl_prompts = [p for p in model.prompts if "Task: make the tests pass" in p]
    assert impl_prompts and not any("== 7" in p for p in impl_prompts)  # implementation never saw the bad test
    assert run.rec.behaviors[0].red_rejections == 1 and run.rec.behaviors[0].green
    assert MUL_RULE.text in model.prompts[0]  # the rule is in the context from the start (show_invariants)


async def test_an_upheld_dispute_rewrites_the_test_without_using_an_attempt_and_then_reaches_green(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([
        BAD_MUL_TEST, dispute("test_b1", "2 times 3 is 6, not 7"), MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL),
    ])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], invariants=[MUL_RULE], settings=WorkflowConfig(check_tests_at_red=False))
    assert run.rec.trace[:9] == [
        "check_assumptions", "write_test", "run_red", "write_impl", "check_dispute", "write_test", "run_red", "write_impl", "run_green",
    ]
    [check] = nodes(run, "check_dispute")
    assert check["upheld"] and check["reason"] == "2 times 3 is 6, not 7" and check["violations"][0]["invariant"] == MUL_RULE.text
    assert "2 times 3 is 6, not 7" in model.prompts[2] and MUL_RULE.text in model.prompts[2]  # the rewrite gets both
    stats = run.rec.behaviors[0]
    assert stats.green and stats.attempts == 1 and (stats.disputes, stats.disputes_upheld) == (1, 1)
    assert "== 6" in (run.ws.path / "tests" / "test_behavior_1.py").read_text()


async def test_a_dispute_that_breaks_no_invariant_is_not_upheld_and_its_reason_reaches_the_next_prompt(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "I believe mul(2, 3) is 5"), MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], invariants=[MUL_RULE])
    [check] = nodes(run, "check_dispute")
    assert not check["upheld"] and check["checked"] == 1 and check["reason"] == "I believe mul(2, 3) is 5"
    assert run.rec.trace[3:8] == ["check_invariants", "write_impl", "check_dispute", "run_green", "write_impl"]
    second = [p for p in model.prompts if "Task: make the tests pass" in p][1]
    assert "not upheld" in second and "I believe mul(2, 3) is 5" in second
    assert "FAILED" in second or "Error" in second  # the suite still ran, so the prompt has its output
    stats = run.rec.behaviors[0]
    assert stats.green and stats.attempts == 2 and (stats.disputes, stats.disputes_upheld) == (1, 0)
    assert nodes(run, "write_impl")[0]["dispute"] == {"test": "test_b1", "reason": "I believe mul(2, 3) is 5"}


async def test_stuck_after_disputes_shows_the_last_reason_and_what_was_checked(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "first reason"), dispute("test_b1", "the test is wrong, it wants 5")])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"], behaviors=["mul multiplies"],
                         assumptions=[], invariants=[MUL_RULE], settings=WorkflowConfig(max_attempts=2))
    [stuck] = run.payloads
    assert stuck["kind"] == "stuck"
    assert stuck["dispute"]["reason"] == "the test is wrong, it wants 5" and not stuck["dispute"]["upheld"]
    assert stuck["dispute"]["checked"] == 1 and stuck["dispute"]["test"] == "test_b1"
    assert run.final["status"] == "aborted"
    shown = io.StringIO()
    show(stuck, shown)
    assert "disputed test_b1: the test is wrong, it wants 5" in shown.getvalue() and "not upheld: 1 checks" in shown.getvalue()


async def test_a_dispute_naming_the_test_file_checks_that_files_examples(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([BAD_MUL_TEST, dispute("tests/test_behavior_1.py", "7 is wrong"), MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], invariants=[MUL_RULE], settings=WorkflowConfig(check_tests_at_red=False))
    assert nodes(run, "check_dispute")[0]["upheld"] and run.rec.behaviors[0].green


async def test_a_dispute_with_no_invariants_in_the_plan_is_not_upheld(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "wrong"), MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"], assumptions=[])
    assert not nodes(run, "check_dispute")[0]["upheld"]
    assert "no Invariant in the Plan" in [p for p in model.prompts if "Task: make the tests pass" in p][1]


async def test_a_dispute_on_an_earlier_behaviors_test_fixes_it_there_and_comes_back(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    wrong_mul = MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a * b + 1\n"  # passes the bad Behavior 1 test
    model = ScriptedModel([
        BAD_MUL_TEST, edit("src/mod.py", wrong_mul),  # Behavior 1 goes Green against a wrong test
        NEG_TEST, dispute("test_b1", "mul(2, 3) is 6", FileEdit(path="src/mod.py", content=NEG)),  # Behavior 2 disputes it
        MUL_TEST,  # Behavior 1's test, corrected
        edit("src/mod.py", NEG),  # Polish
    ])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], invariants=[MUL_RULE], assumptions=[],
                         settings=WorkflowConfig(check_tests_at_red=False))
    trace = run.rec.trace
    i = trace.index("check_dispute")
    assert trace[i:i + 6] == ["check_dispute", "write_test", "run_red", "check_invariants", "resume_behavior", "run_green"]
    rewrite = nodes(run, "write_test")[-1]
    assert rewrite["behavior"] == 0 and "mul(2, 3) is 6" in rewrite["prompt"]
    assert nodes(run, "resume_behavior")[0]["back_to"] == 1
    assert [b.green for b in run.rec.behaviors] == [True, True]
    assert run.rec.behaviors[1].attempts == 0 and run.rec.behaviors[1].disputes_upheld == 1
    assert "== 6" in (run.ws.path / "tests" / "test_behavior_1.py").read_text()


async def test_an_invariant_that_raises_is_reported_as_unchecked_and_the_run_goes_on(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    broken = Invariant(text="a broken rule", check="result.missing")
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, ScriptedModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)]),
                         ["yes", "no"], behaviors=["mul multiplies"], assumptions=[], invariants=[broken])
    [check] = nodes(run, "check_invariants")
    assert check["violations"] == [] and check["unchecked"][0]["invariant"] == "a broken rule"
    assert "AttributeError" in check["unchecked"][0]["error"]
    assert run.rec.behaviors[0].green and run.rec.behaviors[0].red_rejections == 0
