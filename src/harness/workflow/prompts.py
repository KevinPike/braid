"""Prompts for the Execution loop's model nodes. The model sees one Behavior at a time and never the run history."""

from __future__ import annotations

from harness.workflow.plan import Plan

MAX_TREE_LINES = 200
MAX_FILE_LINES = 300

SYSTEM = (
    "You are a careful Python programmer working test-first in a small project. "
    'Reply with JSON only: {"edits": [{"path": ..., "content": ...}], "rationale": ...}. '
    "Each edit is the FULL new contents of one file, path relative to the project root, 1 to 3 files. "
    "No explanations outside the JSON."
)


def trimmed(text: str, limit: int = MAX_FILE_LINES) -> str:
    lines = text.splitlines()
    return text if len(lines) <= limit else "\n".join(lines[:limit]) + f"\n[... {len(lines) - limit} more lines ...]"


def context(plan: Plan, idx: int, tree: list[str], files: dict[str, str]) -> str:
    """The Plan, Behaviors up to the current one, the file tree and the contents of the files that matter.

    Later Behaviors stay hidden: a model that sees them builds them all now, and their tests then pass at Red.
    """
    shown = "\n\n".join(f"--- {rel} ---\n{trimmed(text)}" for rel, text in files.items())
    notes = "".join(f"- {n}\n" for n in plan.notes)
    behaviors = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(plan.behaviors[: idx + 1]))
    tree_text = "\n".join(tree[:MAX_TREE_LINES])
    return (
        f"Goal: {plan.goal}\nOut of scope: {'; '.join(plan.out_of_scope) or 'nothing listed'}\n"
        f"{'Notes:' + chr(10) + notes if notes else ''}"
        f"Behaviors so far, in order:\n{behaviors}\n\nProject files:\n{tree_text}\n\n"
        f"Current file contents:\n{shown or '(none yet)'}"
    )


def write_test(ctx: str, idx: int, behavior: str, hint: str, rejected: str) -> str:
    prompt = (
        f"{ctx}\n\nTask: write ONE new pytest test file, tests/test_behavior_{idx + 1}.py, for this behavior only:\n"
        f"{behavior}\n\nImport from the module as in the existing files; do not change the implementation. "
        "The test must fail until the behavior is implemented."
    )
    if hint:
        prompt += f"\n\nA human hint: {hint}"
    if rejected:
        prompt += f"\n\nYour previous test was rejected. Why, and the output of running it:\n{rejected}"
    return prompt


def vacuous_feedback(output: str) -> str:
    return (
        "The test passes even without any implementation for this behavior: it also passes against the project as it was "
        "before this run started, so it checks nothing new. Write a test that fails until this behavior is implemented.\n"
        f"{output}"
    )


def write_impl(ctx: str, behavior: str, hint: str, last_run: str) -> str:
    prompt = (
        f"{ctx}\n\nTask: make the tests pass by editing the implementation (not the tests). Current behavior: {behavior}\n"
        "Write the SMALLEST change that makes the tests pass. Do not handle cases no test covers yet; later behaviors will "
        "get their own tests. Keep all earlier behaviors working.\n\n"
        f"Last pytest output (tail):\n{last_run}"
    )
    if hint:
        prompt += f"\n\nA human hint: {hint}"
    return prompt


def polish(ctx: str, feedback: str) -> str:
    prompt = (
        f"{ctx}\n\nTask: polish the implementation. Change structure and names (clearer names, type hints, a short "
        "docstring, small helper functions), NEVER behavior. Do not touch tests."
    )
    if feedback:
        prompt += f"\n\nYour last polish was reverted because checks failed:\n{feedback}"
    return prompt
