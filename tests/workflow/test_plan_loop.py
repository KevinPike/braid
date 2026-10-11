"""M14 gates: the Plan loop with a scripted model and a fixture repo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from harness.config import WorkflowConfig
from harness.sandbox import LocalSandbox, Workspace
from harness.workflow import prompts
from harness.workflow.execute import build_execution_graph, drive_graph, initial_state
from harness.workflow.plan import Assumption, Plan
from harness.workflow.plan_loop import (
    ROUND_LIMIT_NOTE, CritiqueReply, PlanDraft, PlanState, Questions, build_plan_graph, initial_plan_state,
)
from harness.workflow.records import Recorder, RunLog
from harness.workflow.research import Finding, Findings, ResearchQueries, ResearchQuery
from tests.workflow.fakes import Reply, SchemaModel
from tests.workflow.test_execute import MUL_IMPL, MUL_TEST, ScriptedModel

GOAL = "add a mul helper"
DEFAULT_ANGLES = ["feasibility", "scope", "testability", "risk", "spacing"]


def read(target: str) -> ResearchQuery:
    return ResearchQuery(kind="read", target=target, why="look")


def draft(*, assumptions: list[Assumption] | None = None, behaviors: list[str] | None = None) -> PlanDraft:
    return PlanDraft(
        behaviors=behaviors or ["mul multiplies two ints", "mul handles zero"],
        assumptions=[Assumption(text="pyproject exists", probe="test -f pyproject.toml")] if assumptions is None else assumptions,
        files=["src/mod.py"], demo="python -c 'from src.mod import mul; print(mul(2, 3))'", out_of_scope=["division"],
    )


def scripted(**over: list[Reply]) -> SchemaModel:
    replies: dict[str, list[Reply]] = {
        "ResearchQueries": [ResearchQueries(queries=[read("src/mod.py")])],
        "Findings": [Findings(findings=[Finding(claim="add sums its arguments", evidence="src/mod.py:2")])],
        "PlanDraft": [draft()],
        "CritiqueReply": [CritiqueReply(points=["fine"])],
        "Questions": [Questions(questions=["Should mul accept floats?"])],
    }
    replies.update(over)
    return SchemaModel(replies)


@dataclass
class Run:
    state: PlanState
    model: SchemaModel
    rec: Recorder
    log: RunLog
    payloads: list[dict[str, Any]]
    plans: Path


async def run_plan(
    project: Path, tmp_path: Path, model: SchemaModel, answers: list[str], settings: WorkflowConfig | None = None,
) -> Run:
    plans = tmp_path / "plans"
    cfg = (settings or WorkflowConfig()).model_copy(update={"plans_dir": plans})
    rec, log = Recorder(), RunLog(tmp_path / "runs" / "plan1.jsonl")
    graph = build_plan_graph(model, project, cfg, plan_id="plan1", recorder=rec, log=log)
    payloads: list[dict[str, Any]] = []

    def ask(payload: dict[str, Any]) -> str:
        payloads.append(payload)
        assert answers, "no scripted answer for the review"
        return answers.pop(0)

    state = await drive_graph(graph, initial_plan_state(GOAL), ask, thread_id="plan")
    return Run(state, model, rec, log, payloads, plans)


def collapsed(trace: list[str]) -> list[str]:
    """The node trace with each run of fanned-out critiques shown once, as ``critique*N``."""
    out: list[str] = []
    for node in trace:
        if node == "critique" and out and out[-1].startswith("critique*"):
            out[-1] = f"critique*{int(out[-1][9:]) + 1}"
        else:
            out.append("critique*1" if node == "critique" else node)
    return out


async def test_one_round_makes_one_call_per_angle_synthesize_once_and_tags_critiques_with_the_round(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["approve"])
    assert len(run.model.of("CritiqueReply")) == 5 and len(run.model.of("Questions")) == 1
    assert sorted(c["angle"] for c in run.state["critiques"]) == sorted(DEFAULT_ANGLES)
    assert {c["round"] for c in run.state["critiques"]} == {1}
    assert collapsed(run.rec.trace) == ["research", "draft", "critique*5", "synthesize", "human_review", "finalize"]
    assert run.state["status"] == "approved" and run.state["round"] == 1


async def test_research_revise_and_approve_each_take_their_route(fixture_repo: Path, tmp_path: Path) -> None:
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["research: how are tests laid out?", "drop the zero case", "approve"])
    assert collapsed(run.rec.trace) == [
        "research", "draft", "critique*5", "synthesize", "human_review",   # round 1: research
        "research", "draft", "critique*5", "synthesize", "human_review",   # round 2: revise
        "draft", "critique*5", "synthesize", "human_review",               # round 3: approve
        "finalize",
    ]
    research_prompts = [c.prompt for c in run.model.of("ResearchQueries")]
    assert "how are tests laid out?" in research_prompts[1] and "how are tests laid out?" not in research_prompts[0]
    drafts = [c.prompt for c in run.model.of("PlanDraft")]
    assert "drop the zero case" in drafts[2] and "drop the zero case" not in drafts[1]
    assert "how are tests laid out?" not in drafts[1]  # a research request is answered by research, not pasted into the draft
    assert "[feasibility] fine" in drafts[1]  # the last round's critiques come along into the redraft
    assert [p["round"] for p in run.payloads] == [1, 2, 3] and run.state["round"] == 3


async def test_the_last_round_finalizes_without_asking_again_and_notes_the_limit(fixture_repo: Path, tmp_path: Path) -> None:
    cfg = WorkflowConfig(max_rounds=2)
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["split it up", "also add logging"], cfg)
    assert len(run.payloads) == 2 and run.state["status"] == "round_limit"
    saved = Plan.load(run.plans / "plan1.json")
    assert ROUND_LIMIT_NOTE in saved.notes and any("also add logging" in n for n in saved.notes)
    # approving on the last round is an approval, not a stop
    approved = await run_plan(fixture_repo, tmp_path / "again", scripted(), ["split it up", "approve"], cfg)
    assert approved.state["status"] == "approved" and ROUND_LIMIT_NOTE not in Plan.load(approved.plans / "plan1.json").notes


async def test_a_query_outside_the_project_is_refused_and_nothing_outside_is_read(fixture_repo: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("top secret contents")
    queries = ResearchQueries(queries=[read("../secret.txt"), read(str(tmp_path / "secret.txt")), read("src/mod.py")])
    run = await run_plan(fixture_repo, tmp_path, scripted(ResearchQueries=[queries]), ["approve"])
    findings_prompt = run.model.of("Findings")[0].prompt
    assert findings_prompt.count("REFUSED") == 2 and "outside the project" in findings_prompt
    assert "top secret" not in "".join(c.prompt for c in run.model.calls)
    [line] = [r for r in run.log.lines() if r["node"] == "research"]
    assert [q["refused"] != "" for q in line["queries"]] == [True, True, False]


async def test_a_finding_with_a_bad_citation_is_unverified_and_becomes_an_open_question(fixture_repo: Path, tmp_path: Path) -> None:
    findings = Findings(findings=[
        Finding(claim="add sums its arguments", evidence="src/mod.py:2"),
        Finding(claim="there is a config loader", evidence="src/config.py:3"),
        Finding(claim="mod has a long docstring", evidence="src/mod.py:400"),
    ])
    run = await run_plan(fixture_repo, tmp_path, scripted(Findings=[findings]), ["approve"])
    assert [f["verified"] for f in run.state["findings"]] == [True, False, False]
    questions = run.state["open_questions"]
    assert any("config loader" in q and "src/config.py:3" in q for q in questions)
    assert any("long docstring" in q and "400" in q for q in questions)
    assert "Should mul accept floats?" in questions  # the model's own questions stay
    assert not any("add sums" in q for q in questions)
    assert Plan.load(run.plans / "plan1.json").open_questions == questions
    assert "(UNVERIFIED" in run.model.of("PlanDraft")[0].prompt  # the draft is told which findings not to lean on


BAD = Assumption(text="the db is reachable", probe=None, evidence=None)
BAD_CITE = Assumption(text="mul belongs in mod", probe=None, evidence="src/mod.py:99")
GOOD_CITE = Assumption(text="add sums", probe=None, evidence="src/mod.py:2 - add returns a + b")


async def test_an_assumption_without_a_probe_or_verified_evidence_triggers_exactly_one_draft_retry(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    fixed = draft(assumptions=[GOOD_CITE, Assumption(text="the db is reachable", probe="true")])
    run = await run_plan(fixture_repo, tmp_path, scripted(PlanDraft=[draft(assumptions=[GOOD_CITE, BAD, BAD_CITE]), fixed]), ["approve"])
    calls = run.model.of("PlanDraft")
    assert len(calls) == 2
    assert "the db is reachable" in calls[1].prompt and "mul belongs in mod" in calls[1].prompt
    assert "the db is reachable" not in calls[0].prompt
    assert run.state["draft_questions"] == [] and not any("Unchecked" in q for q in run.state["open_questions"])

    # still unchecked after the retry: no second retry; the offenders leave the Assumptions and become open questions
    stubborn = draft(assumptions=[GOOD_CITE, BAD])
    run = await run_plan(fixture_repo, tmp_path / "b", scripted(PlanDraft=[stubborn, stubborn]), ["approve"])
    assert len(run.model.of("PlanDraft")) == 2
    plan = Plan.load(run.plans / "plan1.json")
    assert [a.text for a in plan.assumptions] == ["add sums"]
    assert any(q.startswith("Unchecked assumption") and "the db is reachable" in q for q in plan.open_questions)

    clean = await run_plan(fixture_repo, tmp_path / "c", scripted(PlanDraft=[draft(assumptions=[GOOD_CITE])]), ["approve"])
    assert len(clean.model.of("PlanDraft")) == 1


async def test_the_default_critique_angles_include_spacing_and_its_prompt_carries_every_behavior_in_order(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    assert WorkflowConfig().critique_angles == DEFAULT_ANGLES
    behaviors = ["mul multiplies two ints", "mul handles zero", "mul handles negatives", "mul rejects strings"]
    run = await run_plan(fixture_repo, tmp_path, scripted(PlanDraft=[draft(behaviors=behaviors)]), ["approve"])
    [call] = [c for c in run.model.of("CritiqueReply") if c.prompt.startswith("Review angle: spacing")]
    positions = [call.prompt.index(f"{i}. {b}") for i, b in enumerate(behaviors, 1)]
    assert positions == sorted(positions)
    assert call.system == prompts.CRITIQUE_SYSTEMS["spacing"] and "already pass" in call.system


async def test_the_draft_prompt_requires_new_code_per_behavior_and_a_spacing_critique_reaches_synthesize(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    assert "need new code beyond" in prompts.DRAFT_SYSTEM and "fail" in prompts.SPACING_RULE
    spacing_point = "Behavior 2 already passes once Behavior 1 is built"

    def critique(prompt: str) -> CritiqueReply:
        return CritiqueReply(points=[spacing_point] if prompt.startswith("Review angle: spacing") else [])

    run = await run_plan(fixture_repo, tmp_path, scripted(CritiqueReply=[critique]), ["approve"])
    assert {"round": 1, "angle": "spacing", "points": [spacing_point]} in run.state["critiques"]
    assert f"[spacing] {spacing_point}" in run.model.of("Questions")[0].prompt
    assert run.model.of("PlanDraft")[0].system == prompts.DRAFT_SYSTEM
    # the redraft hears about it
    again = await run_plan(fixture_repo, tmp_path / "b", scripted(CritiqueReply=[critique]), ["tighten it", "approve"])
    assert f"[spacing] {spacing_point}" in again.model.of("PlanDraft")[1].prompt


async def test_the_finalized_plan_loads_and_runs_through_the_execution_graph_to_its_first_interrupt(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    one = draft(behaviors=["mul multiplies two ints"])
    run = await run_plan(fixture_repo, tmp_path, scripted(PlanDraft=[one], Questions=[Questions(questions=[])]), ["approve"])
    assert run.state["plan_path"] == str(run.plans / "plan1.json") and run.state["markdown"].startswith(f"# {GOAL}")
    plan = Plan.load(Path(run.state["plan_path"]))
    assert plan.goal == GOAL and plan.behaviors == one.behaviors and plan.open_questions == []

    ws = await Workspace.create(fixture_repo, tmp_path / "workspaces", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    graph = build_execution_graph(ScriptedModel([MUL_TEST, MUL_IMPL]), LocalSandbox(), ws, WorkflowConfig(), hints=True)
    seen: list[str] = []

    class FirstInterrupt(Exception):
        pass

    def ask(payload: dict[str, Any]) -> str:
        seen.append(payload["kind"])
        raise FirstInterrupt

    with pytest.raises(FirstInterrupt):
        await drive_graph(graph, initial_state(plan), ask)
    assert seen == ["sign_off"]  # the Assumption check passed, the Behavior went Red then Green
    await ws.remove()


async def test_a_new_angle_from_config_gets_its_own_prompt_and_every_node_writes_a_run_log_line(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    cfg = WorkflowConfig(critique_angles=["usability"], critique_prompts={"usability": "You are the person who uses the tool."})
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["approve"], cfg)
    [call] = run.model.of("CritiqueReply")
    assert call.system == "You are the person who uses the tool." and call.prompt.startswith("Review angle: usability")
    lines = [ln for ln in run.log.lines() if ln["node"] != "route"]  # Route records are tested in test_trace
    assert [ln["node"] for ln in lines] == run.rec.trace == ["research", "draft", "critique", "synthesize", "human_review", "finalize"]
    assert all("node_seconds" in ln and ln["round"] in (1,) for ln in lines)
    assert lines[1]["prompt"] and lines[1]["reply"]["behaviors"] and lines[-1]["plan"]["goal"] == GOAL
    assert json.loads((run.plans / "plan1.json").read_text())["goal"] == GOAL


def test_markdown_rendering_lists_each_section() -> None:
    plan = Plan(
        goal="g", behaviors=["a", "b"], assumptions=[Assumption(text="t", probe="true", evidence="x.py:1")], files=["x.py"],
        demo="python -c 'print(1)'", out_of_scope=["z"], open_questions=["q?"], notes=["n"],
    )
    md = plan.to_markdown()
    assert md.startswith("# g\n\n## Behaviors\n1. a\n2. b\n") and "(probe: `true`) (evidence: x.py:1)" in md
    for title in ("Files", "Demo", "Out of scope", "Open questions", "Notes"):
        assert f"## {title}" in md
    assert "## Notes" not in Plan(goal="g", behaviors=["a"]).to_markdown()
