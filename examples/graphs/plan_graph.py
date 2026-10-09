"""Plan loop: research, draft, critique from several angles, surface assumptions, ask a human, repeat.

    uv run --no-project --with langgraph python examples/graphs/plan_graph.py [--interactive] [--diagram]

The model is scripted so every run is the same; each place it is called is marked MODEL.
In braid those calls would go through the guard client. The approved plan is written to
plan.json next to this file, and exec_graph.py picks it up.

Graph ideas shown here:
- State is a TypedDict; each node returns only the keys it changes.
- A reducer (Annotated[list, operator.add]) appends updates instead of replacing them,
  which is what lets parallel branches each contribute one item.
- Send() fans out: one critique node per angle runs in the same step, and synthesize
  runs once after all of them finish.
- interrupt() pauses the graph for a human; Command(resume=answer) continues it. This
  needs a checkpointer, because the paused state has to be stored somewhere.
- A conditional edge after the human decides the loop: research again, redraft, or finish.
"""

from __future__ import annotations

import argparse
import json
import operator
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

GOAL = "Export a session transcript to Markdown"
ANGLES = ("feasibility", "scope", "testability", "risk")
MAX_ROUNDS = 3
OUT = Path(__file__).with_name("plan.json")


class Critique(TypedDict):
    round: int
    angle: str
    point: str


class PlanState(TypedDict, total=False):
    goal: str
    round: int
    research: Annotated[list[str], operator.add]
    critiques: Annotated[list[Critique], operator.add]
    feedback: Annotated[list[str], operator.add]
    draft: dict[str, Any]
    open_questions: list[str]


class CritiqueTask(TypedDict):
    """What Send() hands to each critique branch: a private input, not the whole state."""

    round: int
    angle: str
    draft: dict[str, Any]


# --- MODEL (scripted) ---------------------------------------------------------------------
# A real model would get the goal, research, feedback and draft in a prompt and return
# JSON. The fake returns canned answers per round so you can follow the loop.

def model_research(goal: str, feedback: list[str]) -> list[str]:
    if not feedback:
        return [
            "session/store.py keeps messages as Strands dicts: {'role', 'content': [blocks]}",
            "the chat pane already renders Markdown, so fenced blocks match house style",
        ]
    return ["tool results are content blocks shaped {'toolResult': {'content': [{'text': ...}]}}"]


def model_draft(round_: int, feedback: list[str]) -> dict[str, Any]:
    behaviors = ["user and assistant turns render as '## User' / '## Assistant' sections"]
    assumptions: list[dict[str, str | None]] = [
        {"text": "messages are Strands dicts with role + content blocks", "probe": None},
        {"text": "pytest runs in the workspace", "probe": "python -m pytest --version"},
    ]
    if round_ == 1:
        return {
            "behaviors": behaviors + ["the export is written to ~/.braid/exports/<session>.md"],
            "assumptions": assumptions,
            "out_of_scope": [],
        }
    draft: dict[str, Any] = {
        "behaviors": behaviors + ["tool results render as fenced code blocks"],
        "assumptions": assumptions
        + [{"text": "tests can load a real session from fixtures/session.json", "probe": "test -f fixtures/session.json"}],
        "out_of_scope": ["writing files: export_markdown(messages) -> str is pure"],
    }
    if round_ >= 3:
        draft["out_of_scope"].append("image and document blocks are skipped")
    return draft


def model_critique(round_: int, angle: str, draft: dict[str, Any]) -> str:
    script = {
        1: {
            "feasibility": "Writing under ~/.braid needs the write_file approval path (ADR 0003).",
            "scope": "Rendering and file output are two features; start with a pure function.",
            "testability": "A pure export_markdown(messages) -> str is easy to test; file IO is not.",
            "risk": "Unknown: how tool results are shaped in stored messages.",
        },
        2: {
            "feasibility": "One module, no new dependencies.",
            "scope": "Two behaviors is the right size.",
            "testability": "Needs a fixture containing a toolResult block.",
            "risk": "Image blocks would be dropped silently; say whether that is intended.",
        },
    }
    return script.get(round_, {}).get(angle, "No new issues.")


def model_open_questions(round_: int, critiques: list[Critique]) -> list[str]:
    # A real model turns the critiques into questions only a human can answer.
    return {
        1: ["How are tool results stored?", "Write a file, or return a string?"],
        2: ["Image blocks: drop silently, or render a placeholder?"],
    }.get(round_, [])


# --- Nodes -------------------------------------------------------------------------------

def research(state: PlanState) -> dict[str, Any]:
    found = model_research(state["goal"], state.get("feedback", []))
    for line in found:
        print(f"  research  + {line}")
    return {"research": found}


def draft(state: PlanState) -> dict[str, Any]:
    round_ = state.get("round", 0) + 1
    d = model_draft(round_, state.get("feedback", []))
    print(f"  draft     round {round_}: {len(d['behaviors'])} behaviors, {len(d['assumptions'])} assumptions")
    return {"round": round_, "draft": d}


def fan_out(state: PlanState) -> list[Send]:
    return [Send("critique", CritiqueTask(round=state["round"], angle=a, draft=state["draft"])) for a in ANGLES]


def critique(task: CritiqueTask) -> dict[str, Any]:
    point = model_critique(task["round"], task["angle"], task["draft"])
    print(f"  critique  [{task['angle']}] {point}")
    return {"critiques": [Critique(round=task["round"], angle=task["angle"], point=point)]}


def synthesize(state: PlanState) -> dict[str, Any]:
    this_round = [c for c in state["critiques"] if c["round"] == state["round"]]
    return {"open_questions": model_open_questions(state["round"], this_round)}


def human_review(state: PlanState) -> dict[str, Any]:
    # Everything before interrupt() runs again on resume, so keep this side-effect free.
    answer = interrupt(
        {
            "round": state["round"],
            "draft": state["draft"],
            "critiques": [c for c in state["critiques"] if c["round"] == state["round"]],
            "open_questions": state["open_questions"],
        }
    )
    return {"feedback": [str(answer)]}


def after_review(state: PlanState) -> Literal["research", "draft", "finalize"]:
    answer = state["feedback"][-1].strip().lower()
    if answer in {"approve", "ok", "yes", "y"} or state["round"] >= MAX_ROUNDS:
        return "finalize"
    if answer.startswith("research"):
        return "research"
    return "draft"


def finalize(state: PlanState) -> dict[str, Any]:
    plan = {"goal": state["goal"], **state["draft"], "open_questions": state["open_questions"]}
    OUT.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"  finalize  wrote {OUT.name}")
    return {}


def build() -> StateGraph[PlanState]:
    g = StateGraph(PlanState)
    g.add_node("research", research)
    g.add_node("draft", draft)
    g.add_node("critique", critique)
    g.add_node("synthesize", synthesize)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "research")
    g.add_edge("research", "draft")
    g.add_conditional_edges("draft", fan_out, ["critique"])
    g.add_edge("critique", "synthesize")
    g.add_edge("synthesize", "human_review")
    g.add_conditional_edges("human_review", after_review, ["research", "draft", "finalize"])
    g.add_edge("finalize", END)
    return g


# --- Driver ------------------------------------------------------------------------------

SCRIPTED_ANSWERS = [
    "research: how are tool results stored? Also keep it pure: return a string, no file IO.",
    "revise: images are out of scope for now, say so in the plan.",
    "approve",
]


def show_review(payload: dict[str, Any]) -> None:
    d = payload["draft"]
    print(f"\n  ┌─ HUMAN REVIEW (round {payload['round']})")
    for b in d["behaviors"]:
        print(f"  │ behavior:     {b}")
    for a in d["assumptions"]:
        print(f"  │ assumption:   {a['text']}" + (f"   [probe: {a['probe']}]" if a["probe"] else ""))
    for s in d["out_of_scope"]:
        print(f"  │ out of scope: {s}")
    for q in payload["open_questions"]:
        print(f"  │ QUESTION:     {q}")
    print("  └─ approve | research: <what to look into> | <anything else = revise notes>")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interactive", action="store_true", help="answer the reviews yourself")
    p.add_argument("--diagram", action="store_true", help="print the graph as Mermaid and exit")
    args = p.parse_args()

    graph = build().compile(checkpointer=InMemorySaver())
    if args.diagram:
        print(graph.get_graph().draw_mermaid())
        return

    config: Any = {"configurable": {"thread_id": "plan-1"}}
    answers: Iterator[str] = iter(SCRIPTED_ANSWERS)
    run_input: Any = {"goal": GOAL}
    print(f"goal: {GOAL}")
    while True:
        pending = None
        for update in graph.stream(run_input, config, stream_mode="updates"):
            if "__interrupt__" in update:
                pending = update["__interrupt__"][0].value
        if pending is None:
            break
        show_review(pending)
        answer = input("  > ") if args.interactive else next(answers)
        if not args.interactive:
            print(f"  > {answer}")
        print()
        run_input = Command(resume=answer)


if __name__ == "__main__":
    main()
