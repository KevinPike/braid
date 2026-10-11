"""The Plan loop (ADR 0004): research, draft, Critiques fanned out per angle, synthesize, human review, finalize.

Models write, code checks: citations are checked by the harness (``research.py``), an Assumption without a Probe or a
verified citation is bounced, and a human decides when the Plan is done. Node names and routing follow
``examples/graphs/plan_graph.py``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import operator
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypedDict, TypeVar, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from harness.config import WorkflowConfig
from harness.workflow import prompts
from harness.workflow.model import ApiCall, ApiFailure, WorkflowModel, WorkflowModelError
from harness.workflow.invariants import Invariant
from harness.workflow.plan import Assumption, Plan
from harness.workflow.records import Recorder, RunLog
from harness.workflow.research import check_citation, research, summarize

T = TypeVar("T", bound=BaseModel)

APPROVE = {"approve", "approved", "ok", "yes", "y", "lgtm", "looks good"}
ROUND_LIMIT_NOTE = "stopped at round limit"
MAX_QUESTIONS = 5
MAX_POINTS = 3


class PlanDraft(BaseModel):
    """What the ``draft`` node asks the model for: a Plan without the parts only the loop fills in."""

    behaviors: list[str] = Field(min_length=1)
    assumptions: list[Assumption] = []
    files: list[str] = []
    demo: str | None = None
    out_of_scope: list[str] = []
    invariants: list[Invariant] = []


class CritiqueReply(BaseModel):
    points: list[str] = Field(max_length=MAX_POINTS)


class Questions(BaseModel):
    questions: list[str] = Field(max_length=MAX_QUESTIONS)


class CritiqueRecord(TypedDict):
    round: int
    angle: str
    points: list[str]


class PlanState(TypedDict, total=False):
    goal: str
    round: int  # the draft being reviewed; 0 before the first draft
    findings: Annotated[list[dict[str, Any]], operator.add]  # CheckedFinding dumps, each with the round it was found for
    research_questions: Annotated[list[str], operator.add]
    critiques: Annotated[list[CritiqueRecord], operator.add]  # reducer: parallel branches append; each is tagged with its round
    feedback: Annotated[list[str], operator.add]  # the human's answers, in order
    draft: dict[str, Any]  # PlanDraft.model_dump()
    draft_questions: list[str]  # Assumptions dropped for want of a Probe or a verified citation
    open_questions: list[str]
    plan_path: str
    markdown: str
    status: str
    api_error: dict[str, Any]  # F8: set by drive_graph on a status "api_error" result, never checkpointed
    recorder: dict[str, Any]  # Recorder.dump() after each node


class CritiqueTask(TypedDict):
    """What ``Send`` hands each critique branch: a private input, not the whole state."""

    round: int
    angle: str
    goal: str
    draft: dict[str, Any]


class _NodeFn(Protocol):
    async def __call__(self, state: PlanState) -> dict[str, Any]: ...


class _SerialModel:
    """One model call at a time: Critiques run side by side in the graph, but a local GPU serves one request, and the
    timings stay honest (latency excludes waiting for a sibling). It also hands each node the API calls it made."""

    def __init__(self, model: WorkflowModel, rec: Recorder) -> None:
        self.model, self.rec = model, rec
        self.lock = asyncio.Lock()
        self.used: dict[asyncio.Task[Any] | None, list[ApiCall]] = {}  # per node (its task), until the node logs

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        return (await self.timed(system, prompt, schema))[0]

    async def timed(self, system: str, prompt: str, schema: type[T]) -> tuple[T, float]:
        async with self.lock:
            calls = getattr(self.model, "calls", None)
            mark = len(calls) if isinstance(calls, list) else 0
            start = time.monotonic()
            try:
                return await self.model.complete(system, prompt, schema), time.monotonic() - start
            finally:
                self.rec.model_calls += 1
                self.rec.model_seconds += time.monotonic() - start
                if isinstance(calls, list):
                    self.used.setdefault(asyncio.current_task(), []).extend(c for c in calls[mark:] if isinstance(c, ApiCall))

    def take(self) -> list[ApiCall]:
        """The API calls the current node made since it last took."""
        return self.used.pop(asyncio.current_task(), [])


def plan_text(draft: dict[str, Any]) -> str:
    """The draft as a model reads it: Behaviors numbered in order, then Assumptions, files, demo and scope."""
    d = PlanDraft.model_validate(draft)
    lines = ["Behaviors, in order:", *(f"{i}. {b}" for i, b in enumerate(d.behaviors, 1))]
    lines += ["Assumptions:", *(f"- {a.text} | probe: {a.probe or '-'} | evidence: {a.evidence or '-'}" for a in d.assumptions)]
    lines += [f"Files: {', '.join(d.files) or '-'}", f"Demo: {d.demo or '-'}", f"Out of scope: {'; '.join(d.out_of_scope) or '-'}"]
    return "\n".join(lines)


def is_approval(answer: str) -> bool:
    return answer.strip().lower().rstrip(".!") in APPROVE


def research_text(answer: str) -> str | None:
    """The text after ``research:`` in a human answer, or None when the answer is not a research request."""
    stripped = answer.strip()
    if not stripped.lower().startswith("research"):
        return None
    return stripped[len("research"):].lstrip(" :-").strip()


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i.strip()))


class _PlanLoop:
    def __init__(
        self, model: WorkflowModel, project: Path, settings: WorkflowConfig, *, plan_id: str, recorder: Recorder, log: RunLog | None,
    ) -> None:
        self.model = _SerialModel(model, recorder)
        self.project, self.cfg, self.plan_id, self.rec, self.runlog = project, settings, plan_id, recorder, log

    def log(self, node: str, started: float, round_: int, **fields: Any) -> None:
        self.rec.trace.append(node)
        usage = self.rec.add_usage(self.model.take())
        if self.runlog:
            self.runlog.write(node, round=round_, node_seconds=round(time.monotonic() - started, 2), **fields, **usage)

    # --- research ------------------------------------------------------------------------------------------
    async def research(self, state: PlanState) -> dict[str, Any]:
        started = time.monotonic()
        round_ = state.get("round", 0) + 1  # the draft this research is for
        asked = research_text(state["feedback"][-1]) if state.get("feedback") else None
        known = [f["claim"] for f in state.get("findings", []) if f["verified"]]
        before = self.rec.model_seconds
        out = await research(self.model, self.project, state["goal"], feedback=asked or "", known=known)
        self.log(
            "research", started, round_, prompts=out.prompts, queries=summarize(out.results),
            findings=[f.model_dump() for f in out.findings], questions=out.questions, model_seconds=round(self.rec.model_seconds - before, 2),
        )
        return {
            "findings": [{**f.model_dump(), "round": round_} for f in out.findings],
            "research_questions": out.questions,
        }

    # --- draft ---------------------------------------------------------------------------------------------
    def unchecked(self, draft: PlanDraft) -> list[Assumption]:
        """Assumptions with neither a Probe nor a citation that the harness can verify."""
        return [
            a for a in draft.assumptions
            if not (a.probe and a.probe.strip()) and not (a.evidence and not check_citation(self.project, a.evidence))
        ]

    async def draft(self, state: PlanState) -> dict[str, Any]:
        started = time.monotonic()
        round_ = state.get("round", 0) + 1
        findings = [prompts.verified_line(f["claim"], f["evidence"], f["verified"]) for f in state.get("findings", [])]
        previous = plan_text(state["draft"]) if state.get("draft") else ""
        answer = state["feedback"][-1] if state.get("feedback") else ""
        feedback = "" if research_text(answer) is not None else answer  # a research request was answered by research
        last = [f"[{c['angle']}] {p}" for c in state.get("critiques", []) if c["round"] == round_ - 1 for p in c["points"]]
        prompt = prompts.draft(state["goal"], findings, previous, feedback, last, "")
        draft, latency = await self.model.timed(prompts.DRAFT_SYSTEM, prompt, PlanDraft)
        retried = False
        offending = self.unchecked(draft)
        if offending:  # exactly one retry, naming them
            retried = True
            prompt = prompts.draft(state["goal"], findings, previous, feedback, last, prompts.retry_unchecked([a.text for a in offending]))
            draft, again = await self.model.timed(prompts.DRAFT_SYSTEM, prompt, PlanDraft)
            latency += again
            offending = self.unchecked(draft)
        questions: list[str] = []
        if offending:
            dropped = {id(a) for a in offending}
            draft = draft.model_copy(update={"assumptions": [a for a in draft.assumptions if id(a) not in dropped]})
            questions = [f"Unchecked assumption (no probe, no verified evidence): {a.text}" for a in offending]
        self.log(
            "draft", started, round_, prompt=prompt, reply=draft.model_dump(), retried=retried, dropped=questions,
            model_seconds=round(latency, 2),
        )
        return {"round": round_, "draft": draft.model_dump(), "draft_questions": questions}

    # --- critique, fanned out ------------------------------------------------------------------------------
    def fan_out(self, state: PlanState) -> list[Send] | Literal["synthesize"]:
        angles = self.cfg.critique_angles
        if not angles:
            return "synthesize"
        return [Send("critique", CritiqueTask(round=state["round"], angle=a, goal=state["goal"], draft=state["draft"])) for a in angles]

    async def critique(self, state: CritiqueTask) -> dict[str, Any]:  # "state" is the task Send handed over
        task = state
        started = time.monotonic()
        system = prompts.critique_system(task["angle"], self.cfg.critique_prompts)
        prompt = prompts.critique(task["angle"], task["goal"], plan_text(task["draft"]))
        try:
            reply, latency = await self.model.timed(system, prompt, CritiqueReply)
            points, error = reply.points, ""
        except WorkflowModelError as exc:  # one angle failing must not sink the round
            latency, points, error = 0.0, [], str(exc)
        self.log(
            "critique", started, task["round"], angle=task["angle"], prompt=prompt, points=points, model_error=error,
            model_seconds=round(latency, 2),
        )
        return {"critiques": [CritiqueRecord(round=task["round"], angle=task["angle"], points=points)]}

    # --- synthesize ----------------------------------------------------------------------------------------
    async def synthesize(self, state: PlanState) -> dict[str, Any]:
        started = time.monotonic()
        this_round = [(c["angle"], p) for c in state.get("critiques", []) if c["round"] == state["round"] for p in c["points"]]
        prompt = prompts.synthesize(state["goal"], plan_text(state["draft"]), this_round)
        try:
            reply, latency = await self.model.timed(prompts.SYNTHESIZE_SYSTEM, prompt, Questions)
            asked, error = reply.questions, ""
        except WorkflowModelError as exc:
            latency, asked, error = 0.0, [], str(exc)
        # what the harness found itself comes along: unverified findings and Assumptions dropped from the draft
        questions = dedupe([*asked, *state.get("research_questions", []), *state.get("draft_questions", [])])
        self.log("synthesize", started, state["round"], prompt=prompt, asked=asked, questions=questions, model_error=error, model_seconds=round(latency, 2))
        return {"open_questions": questions}

    # --- human review --------------------------------------------------------------------------------------
    async def human_review(self, state: PlanState) -> dict[str, Any]:
        started = time.monotonic()
        # everything before interrupt() runs again on resume, so it stays free of side effects
        answer = str(interrupt({
            "kind": "review", "round": state["round"], "max_rounds": self.cfg.max_rounds, "goal": state["goal"],
            "draft": state["draft"], "open_questions": state.get("open_questions", []),
            "critiques": [c for c in state.get("critiques", []) if c["round"] == state["round"]],
            "unverified": [f for f in state.get("findings", []) if not f["verified"]],
        }))
        route = "approve" if is_approval(answer) else "research" if research_text(answer) is not None else "revise"
        self.log("human_review", started, state["round"], answer=answer, route=route)
        return {"feedback": [answer]}

    def after_review(self, state: PlanState) -> Literal["research", "draft", "finalize"]:
        answer = state["feedback"][-1]
        if is_approval(answer) or state["round"] >= self.cfg.max_rounds:
            return "finalize"
        return "research" if research_text(answer) is not None else "draft"

    # --- finalize ------------------------------------------------------------------------------------------
    async def finalize(self, state: PlanState) -> dict[str, Any]:
        started = time.monotonic()
        answer = state["feedback"][-1]
        notes: list[str] = []
        status = "approved"
        if not is_approval(answer):  # the round limit ended the loop with the human's last words unapplied
            status = "round_limit"
            notes = [ROUND_LIMIT_NOTE, f"unapplied review feedback: {answer.strip()}"]
        plan = Plan(goal=state["goal"], **state["draft"], open_questions=state.get("open_questions", []), notes=notes)
        path = self.cfg.plans_dir.expanduser() / f"{self.plan_id}.json"
        plan.save(path)
        markdown = plan.to_markdown()
        self.log("finalize", started, state["round"], outcome=status, path=str(path), plan=json.loads(plan.model_dump_json()))
        return {"plan_path": str(path), "markdown": markdown, "status": status}

    # --- Spend cap -----------------------------------------------------------------------------------------
    def spend_cap(self, state: PlanState) -> dict[str, Any]:
        """Terminal node: the run cost more than ``max_cost_usd``; no Plan is saved."""
        self.log("spend_cap", time.monotonic(), state.get("round", 0), spent=round(self.rec.cost_usd, 6), cap=self.cfg.max_cost_usd)
        return {"status": "spend_cap"}

    def gate(self, route: Callable[[PlanState], Any], *, let: str = "") -> Callable[[PlanState], Any]:
        """``route``, unless the Spend cap has been passed. ``let`` is a destination that still runs: finalize makes no
        model call, so a Plan the human has already approved is saved."""
        @functools.wraps(route)
        def gated(state: PlanState) -> Any:
            where = route(state)
            return where if where == let or self.rec.cost_usd <= self.cfg.max_cost_usd else "spend_cap"

        return gated

    def timed(self, node: Callable[[PlanState], Any]) -> _NodeFn:
        async def run(state: PlanState) -> dict[str, Any]:
            try:
                result = node(state)
                update = cast(dict[str, Any], await result if inspect.isawaitable(result) else result)
            except ApiFailure as exc:  # F8: logged here, then ``drive_graph`` ends the run with status api_error
                self.log("api_error", time.monotonic(), state.get("round", 0), at=node.__name__, **exc.record())
                raise
            return {**update, "recorder": self.rec.dump()}

        return run

    def graph(self) -> StateGraph[PlanState]:
        g = StateGraph(PlanState)
        for name in ("research", "draft", "synthesize", "human_review", "finalize", "spend_cap"):
            g.add_node(name, self.timed(getattr(self, name)))
        g.add_node("critique", self.critique, input_schema=CritiqueTask)  # its input is the Send payload, not the state

        def then(name: str) -> Callable[[PlanState], str]:
            return lambda _state: name

        g.add_edge(START, "research")
        g.add_conditional_edges("research", self.gate(then("draft")), ["draft", "spend_cap"])
        g.add_conditional_edges("draft", self.gate(self.fan_out), ["critique", "synthesize", "spend_cap"])
        g.add_edge("critique", "synthesize")
        g.add_conditional_edges("synthesize", self.gate(then("human_review")), ["human_review", "spend_cap"])
        g.add_conditional_edges("human_review", self.gate(self.after_review, let="finalize"), ["research", "draft", "finalize", "spend_cap"])
        g.add_edge("spend_cap", END)
        g.add_edge("finalize", END)
        return g


def initial_plan_state(goal: str) -> PlanState:
    return {"goal": goal, "round": 0}


def build_plan_graph(
    model: WorkflowModel, project: Path, settings: WorkflowConfig, *, plan_id: str, recorder: Recorder | None = None,
    log: RunLog | None = None,
) -> StateGraph[PlanState]:
    """The Plan loop as a LangGraph ``StateGraph``. ``finalize`` saves the Plan to ``settings.plans_dir/<plan_id>.json``
    and leaves its Markdown in the state; ``recorder`` and ``log`` collect the trace and one JSONL line per node."""
    return _PlanLoop(model, project, settings, plan_id=plan_id, recorder=recorder or Recorder(), log=log).graph()
