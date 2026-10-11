"""Execution loop: check the Plan's Assumptions, make it work test-first, get Sign-off, then Polish.

    uv run --no-project --with langgraph --with pytest python examples/graphs/exec_graph.py [--interactive] [--diagram]

Reads plan.json from plan_graph.py (or a built-in copy of it). The model is scripted and
marked MODEL; everything else is real: probes run as shell commands and pytest runs in a
throwaway workspace. The scripted model gets behavior 2 wrong once, so the retry loop shows.

The three stages are separate parts of the graph, and each one can only be entered from
the one before (terms as in CONTEXT.md):

    assumption check  check_assumptions <-> clarify          (stop before writing code on a bad assumption)
    make it work      write_test -> run_red -> write_impl -> run_green -> next_behavior   (per behavior)
    make it right     sign_off (human) -> polish -> verify_polish -> done   (tests must stay green, else revert)
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MAX_ATTEMPTS = 3
PLAN = Path(__file__).with_name("plan.json")
IMPL = "markdown_export.py"
DEFAULT_PLAN: dict[str, Any] = {
    "goal": "Export a session transcript to Markdown",
    "behaviors": [
        "user and assistant turns render as '## User' / '## Assistant' sections",
        "tool results render as fenced code blocks",
    ],
    "assumptions": [
        {"text": "messages are Strands dicts with role + content blocks", "probe": None},
        {"text": "pytest runs in the workspace", "probe": "python -m pytest --version"},
        {"text": "tests can load a real session from fixtures/session.json", "probe": "test -f fixtures/session.json"},
    ],
    "out_of_scope": ["writing files", "image and document blocks are skipped"],
    "open_questions": [],
}


class ExecState(TypedDict, total=False):
    plan: dict[str, Any]
    workdir: str
    waived: Annotated[list[str], operator.add]
    failed: list[str]
    idx: int
    attempts: int
    hint: str
    last_run: str
    passed: bool
    backup: str
    status: str


# --- MODEL (scripted) ---------------------------------------------------------------------
# A real model gets the behavior, the current file and the trimmed pytest output, and
# returns one file's full contents as JSON. No tool calling, so a small model can do it.

TESTS = [
    '''from markdown_export import export_markdown


def test_turns_render_as_sections():
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "hello"}]},
    ]
    assert export_markdown(messages) == "## User\\n\\nhi\\n\\n## Assistant\\n\\nhello"
''',
    '''from markdown_export import export_markdown


def test_tool_results_are_fenced():
    messages = [{"role": "user", "content": [{"toolResult": {"content": [{"text": "a.txt\\nb.txt"}]}}]}]
    assert export_markdown(messages) == "## User\\n\\n```\\na.txt\\nb.txt\\n```"
''',
]

IMPL_TURNS = '''def export_markdown(messages):
    parts = []
    for m in messages:
        parts.append("## " + m["role"].title())
        for block in m["content"]:
            if "text" in block:
                parts.append(block["text"])
    return "\\n\\n".join(parts)
'''

IMPL_TOOLS_WRONG = IMPL_TURNS.replace(
    '''                parts.append(block["text"])
''',
    '''                parts.append(block["text"])
            if "toolResult" in block:
                for inner in block["toolResult"]["content"]:
                    parts.append(inner["text"])
''',
)

IMPL_TOOLS = IMPL_TOOLS_WRONG.replace('parts.append(inner["text"])', 'parts.append("```\\n" + inner["text"] + "\\n```")')

POLISHED = '''"""Render a Strands message list as Markdown."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

Block = dict[str, Any]


def export_markdown(messages: Iterable[Block]) -> str:
    """One `## Role` section per message; tool results become fenced blocks."""
    sections: list[str] = []
    for message in messages:
        sections.append(f"## {message['role'].title()}")
        for block in message["content"]:
            sections.extend(_render(block))
    return "\\n\\n".join(sections)


def _render(block: Block) -> list[str]:
    if "text" in block:
        return [block["text"]]
    if "toolResult" in block:
        return [_fence(item["text"]) for item in block["toolResult"]["content"] if "text" in item]
    return []  # images and documents are out of scope


def _fence(text: str) -> str:
    return f"```\\n{text}\\n```"
'''


def model_write_test(behavior: str, idx: int) -> str:
    if idx >= len(TESTS):
        sys.exit(f"\nThe scripted model only knows the planned behaviors; a real model would write a test for: {behavior}")
    return TESTS[idx]


def model_write_impl(behavior: str, idx: int, attempt: int, current: str, last_run: str, hint: str) -> str:
    if idx == 0:
        return IMPL_TURNS
    return IMPL_TOOLS_WRONG if attempt == 1 and not hint else IMPL_TOOLS


def model_polish(current: str, plan: dict[str, Any]) -> str:
    return POLISHED


# --- Workspace ---------------------------------------------------------------------------

def sh(cmd: str, cwd: str) -> tuple[int, str]:
    # Put this interpreter first on PATH so "python" in a probe means the env that has pytest.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def pytest(cwd: str, target: str = "tests") -> tuple[bool, str]:
    code, out = sh(f"python -m pytest -q --no-header -p no:cacheprovider {target}", cwd)
    tail = "\n".join(out.splitlines()[-12:])  # small models get a trimmed failure, not the whole log
    return code == 0, tail


def write(state: ExecState, rel: str, text: str) -> None:
    path = Path(state["workdir"], rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read(state: ExecState, rel: str) -> str:
    path = Path(state["workdir"], rel)
    return path.read_text() if path.exists() else ""


def behavior(state: ExecState) -> str:
    return str(state["plan"]["behaviors"][state["idx"]])


# --- Stage 1: assumption check ---------------------------------------------------------------

def check_assumptions(state: ExecState) -> dict[str, Any]:
    failed = []
    for a in state["plan"]["assumptions"]:
        if a["text"] in state.get("waived", []):
            print(f"  check     ~ {a['text']}  (waived)")
        elif a["probe"] is None:
            print(f"  check     ✓ {a['text']}  (verified during planning)")
        else:
            code, _ = sh(a["probe"], state["workdir"])
            print(f"  check     {'✓' if code == 0 else '✗'} {a['text']}  [{a['probe']}]")
            if code != 0:
                failed.append(a["text"])
    return {"failed": failed}


def after_check(state: ExecState) -> Literal["clarify", "write_test"]:
    return "clarify" if state["failed"] or state["plan"]["open_questions"] else "write_test"


def clarify(state: ExecState) -> dict[str, Any]:
    answer = str(interrupt({"kind": "clarify", "failed": state["failed"], "open_questions": state["plan"]["open_questions"]}))
    if answer.lower().startswith("abort"):
        return {"status": "aborted"}
    # A real version would let the model fold the answer into the plan; here it waives and notes it.
    plan = {**state["plan"], "open_questions": [], "notes": [*state["plan"].get("notes", []), answer]}
    return {"waived": state["failed"], "plan": plan}


def after_clarify(state: ExecState) -> Literal["check_assumptions", "__end__"]:
    return END if state.get("status") == "aborted" else "check_assumptions"


# --- Stage 2: make it work ---------------------------------------------------------------

def write_test(state: ExecState) -> dict[str, Any]:
    idx = state.get("idx", 0)
    rel = f"tests/test_behavior_{idx + 1}.py"
    write(state, rel, model_write_test(state["plan"]["behaviors"][idx], idx))
    print(f"\n  write_test  behavior {idx + 1}: {state['plan']['behaviors'][idx]}")
    return {"idx": idx, "attempts": 0, "hint": ""}


def run_red(state: ExecState) -> dict[str, Any]:
    passed, out = pytest(state["workdir"], f"tests/test_behavior_{state['idx'] + 1}.py")
    print(f"  run_red     {'PASSED (test is not testing anything new)' if passed else 'failed, as it should'}")
    return {"passed": passed, "last_run": out}


def after_red(state: ExecState) -> Literal["write_test", "write_impl"]:
    return "write_test" if state["passed"] else "write_impl"


def write_impl(state: ExecState) -> dict[str, Any]:
    attempt = state["attempts"] + 1
    code = model_write_impl(behavior(state), state["idx"], attempt, read(state, IMPL), state["last_run"], state["hint"])
    write(state, IMPL, code)
    print(f"  write_impl  attempt {attempt}")
    return {"attempts": attempt}


def run_green(state: ExecState) -> dict[str, Any]:
    passed, out = pytest(state["workdir"])  # the whole suite, so earlier behaviors cannot regress
    print(f"  run_green   {'all tests pass' if passed else 'FAILED'}")
    if not passed:
        print("\n".join(f"              | {line}" for line in out.splitlines()[-4:]))
    return {"passed": passed, "last_run": out}


def after_green(state: ExecState) -> Literal["next_behavior", "write_impl", "stuck"]:
    if state["passed"]:
        return "next_behavior"
    return "write_impl" if state["attempts"] < MAX_ATTEMPTS else "stuck"


def stuck(state: ExecState) -> dict[str, Any]:
    answer = str(interrupt({"kind": "stuck", "behavior": behavior(state), "last_run": state["last_run"]}))
    if answer.lower().startswith("abort"):
        return {"status": "aborted"}
    return {"attempts": 0, "hint": answer}


def after_stuck(state: ExecState) -> Literal["write_impl", "__end__"]:
    return END if state.get("status") == "aborted" else "write_impl"


def next_behavior(state: ExecState) -> dict[str, Any]:
    return {"idx": state["idx"] + 1}


def after_next(state: ExecState) -> Literal["write_test", "sign_off"]:
    return "write_test" if state["idx"] < len(state["plan"]["behaviors"]) else "sign_off"


# --- Stage 3: make it right -------------------------------------------------------------

DEMO = (
    "from markdown_export import export_markdown; print(export_markdown(["
    "{'role': 'user', 'content': [{'text': 'list the files'}]},"
    "{'role': 'assistant', 'content': [{'toolResult': {'content': [{'text': 'a.txt\\nb.txt'}]}}, {'text': 'Two files.'}]}]))"
)


def sign_off(state: ExecState) -> dict[str, Any]:
    _, demo = sh(f'python -c "{DEMO}"', state["workdir"])
    answer = str(interrupt({"kind": "sign_off", "demo": demo}))
    if answer.lower().startswith(("y", "ok", "looks good")):
        return {}
    # "No, and also X" becomes a new behavior, so the work loop runs again.
    return {"plan": {**state["plan"], "behaviors": [*state["plan"]["behaviors"], answer]}}


def after_sign_off(state: ExecState) -> Literal["polish", "write_test"]:
    return "write_test" if state["idx"] < len(state["plan"]["behaviors"]) else "polish"


def polish(state: ExecState) -> dict[str, Any]:
    backup = read(state, IMPL)
    write(state, IMPL, model_polish(backup, state["plan"]))
    print("\n  polish      refactor for names, types and structure; behavior must not change")
    return {"backup": backup}


def verify_polish(state: ExecState) -> dict[str, Any]:
    # A real version would also run mypy/ruff here and loop back to polish on findings.
    passed, out = pytest(state["workdir"])
    if passed:
        print("  verify      tests still pass, keeping the refactor")
        return {"status": "done"}
    write(state, IMPL, state["backup"])
    print("  verify      refactor broke a test, reverted to the working version")
    return {"status": "done", "last_run": out}


def build() -> StateGraph[ExecState]:
    g = StateGraph(ExecState)
    for name, fn in [
        ("check_assumptions", check_assumptions), ("clarify", clarify),
        ("write_test", write_test), ("run_red", run_red), ("write_impl", write_impl),
        ("run_green", run_green), ("stuck", stuck), ("next_behavior", next_behavior),
        ("sign_off", sign_off), ("polish", polish), ("verify_polish", verify_polish),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "check_assumptions")
    g.add_conditional_edges("check_assumptions", after_check, ["clarify", "write_test"])
    g.add_conditional_edges("clarify", after_clarify, ["check_assumptions", END])
    g.add_edge("write_test", "run_red")
    g.add_conditional_edges("run_red", after_red, ["write_test", "write_impl"])
    g.add_edge("write_impl", "run_green")
    g.add_conditional_edges("run_green", after_green, ["next_behavior", "write_impl", "stuck"])
    g.add_conditional_edges("stuck", after_stuck, ["write_impl", END])
    g.add_conditional_edges("next_behavior", after_next, ["write_test", "sign_off"])
    g.add_conditional_edges("sign_off", after_sign_off, ["polish", "write_test"])
    g.add_edge("polish", "verify_polish")
    g.add_edge("verify_polish", END)
    return g


# --- Driver ------------------------------------------------------------------------------

SCRIPTED_ANSWERS = iter([
    "waive it: build message fixtures inline in each test instead",
    "yes",
])


def show(payload: dict[str, Any]) -> None:
    print(f"\n  ┌─ HUMAN: {payload['kind']}")
    if payload["kind"] == "clarify":
        for f in payload["failed"]:
            print(f"  │ assumption failed: {f}")
        for q in payload["open_questions"]:
            print(f"  │ open question:     {q}")
        print("  └─ <how to proceed> | abort")
    elif payload["kind"] == "stuck":
        print(f"  │ {MAX_ATTEMPTS} attempts failed on: {payload['behavior']}")
        print("\n".join(f"  │   {line}" for line in payload["last_run"].splitlines()[-6:]))
        print("  └─ <a hint for the model> | abort")
    else:
        print("  │ demo output:")
        print("\n".join(f"  │   {line}" for line in payload["demo"].splitlines()))
        print("  └─ yes | <what's wrong or missing; becomes a new behavior>")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interactive", action="store_true", help="answer the pauses yourself")
    p.add_argument("--diagram", action="store_true", help="print the graph as Mermaid and exit")
    args = p.parse_args()

    graph = build().compile(checkpointer=InMemorySaver())
    if args.diagram:
        print(graph.get_graph().draw_mermaid())
        return

    plan = json.loads(PLAN.read_text()) if PLAN.exists() else DEFAULT_PLAN
    workdir = tempfile.mkdtemp(prefix="exec-graph-")
    print(f"goal: {plan['goal']}\nworkspace: {workdir}\n")
    config: Any = {"configurable": {"thread_id": "exec-1"}, "recursion_limit": 100}
    run_input: Any = {"plan": plan, "workdir": workdir}
    while True:
        pending = None
        for update in graph.stream(run_input, config, stream_mode="updates"):
            if "__interrupt__" in update:
                pending = update["__interrupt__"][0].value
        if pending is None:
            break
        show(pending)
        answer = input("  > ") if args.interactive else next(SCRIPTED_ANSWERS)
        if not args.interactive:
            print(f"  > {answer}")
        run_input = Command(resume=answer)

    final = graph.get_state(config).values
    print(f"\nstatus: {final.get('status')}\n\n--- {IMPL} ---\n{read(final, IMPL)}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
