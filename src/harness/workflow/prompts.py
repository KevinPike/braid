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


# --- Plan loop (M14) ------------------------------------------------------------------------------------------------

RESEARCH_SYSTEM = (
    "You research a small Python project so a plan can be grounded in what the code really does. "
    "You cannot run tools: you name what to look at and the harness reads it for you. Reply with JSON only."
)
FINDINGS_SYSTEM = (
    "You turn what was read from a project into short, checkable findings. Every finding cites one place as "
    '"path:line" (a line you were shown). Never cite a file or line you were not shown. Reply with JSON only.'
)


def research_queries(goal: str, tree: list[str], feedback: str, known: list[str]) -> str:
    prompt = (
        f"Goal: {goal}\n\nProject files:\n{chr(10).join(tree[:MAX_TREE_LINES]) or '(none yet)'}\n\n"
        "Ask for at most 6 lookups that would tell you what the plan must know before it is written. Each is "
        '{"kind": "read" | "grep" | "list", "target": ..., "why": ...}: read = a file path (first 300 lines), '
        "grep = a Python regular expression searched in every tracked file (first 50 hits), list = a directory path. "
        "Paths are relative to the project root. If the project is empty or new, ask for few or no lookups."
    )
    if known:
        prompt += "\n\nAlready found (do not look these up again):\n" + "\n".join(f"- {k}" for k in known)
    if feedback:
        prompt += f"\n\nThe human asked you to look into: {feedback}"
    return prompt


def research_findings(goal: str, results: str) -> str:
    return (
        f"Goal: {goal}\n\nWhat the lookups returned (lines are numbered):\n{results}\n\n"
        'Reply with {"findings": [{"claim": ..., "evidence": "path:line"}]}: facts about the project that matter for '
        "this goal, each cited to a line shown above. Refused or empty lookups give no findings."
    )


def verified_line(claim: str, evidence: str, verified: bool) -> str:
    return f"- {claim} [{evidence}]" if verified else f"- (UNVERIFIED, do not rely on it) {claim} [{evidence}]"


SPACING_RULE = (
    "Each Behavior must need new code beyond what the earlier Behaviors already required: a test for it must fail "
    "against an implementation that only satisfies the Behaviors before it. Do not list a Behavior that earlier code "
    "would already cover, and do not hide two steps in one Behavior."
)

DRAFT_SYSTEM = (
    "You write a Plan for a small Python feature that will be built test-first by a small local model. Reply with JSON "
    "only, in this shape: behaviors (3 to 6 short sentences, simplest first, each provable by ONE test and about 30 lines "
    "of implementation), assumptions (each {text, probe, evidence}), files (project paths the work reads or touches), "
    "demo (a `python -c` command that prints the feature's output on the goal's examples, or null), out_of_scope. "
    f"{SPACING_RULE} "
    "Every assumption needs either a probe, a shell command that exits 0 when the assumption holds and needs no network, "
    "or evidence copied from a VERIFIED finding as path:line. Prefer a probe; never invent evidence."
)


def draft(goal: str, findings: list[str], previous: str, feedback: str, critiques: list[str], retry: str) -> str:
    prompt = f"Goal: {goal}\n\nVerified and unverified findings from research:\n{chr(10).join(findings) or '(none)'}"
    if previous:
        prompt += f"\n\nYour previous draft:\n{previous}"
    if critiques:
        prompt += "\n\nCritiques of that draft:\n" + "\n".join(f"- {c}" for c in critiques)
    if feedback:
        prompt += f"\n\nThe human's review of that draft (apply it): {feedback}"
    if retry:
        prompt += f"\n\n{retry}"
    return prompt


def retry_unchecked(texts: list[str]) -> str:
    listed = "\n".join(f"- {t}" for t in texts)
    return (
        "Your last draft had assumptions with neither a probe nor evidence from a verified finding:\n"
        f"{listed}\nGive each one a probe (a shell command that exits 0 when it holds) or evidence cited from a verified "
        "finding, or remove it. Return the whole plan again."
    )


CRITIQUE_SYSTEMS: dict[str, str] = {
    "feasibility": (
        "You review a draft Plan for feasibility. Could a small local model build this with the project and tools "
        "described? Name what is unrealistic, missing or unknown. Reply with JSON {\"points\": [...]}, at most 3 short points."
    ),
    "scope": (
        "You review a draft Plan for scope. Is it too big, or does it hide extra features? Name what to cut or move to "
        "out of scope. Reply with JSON {\"points\": [...]}, at most 3 short points."
    ),
    "testability": (
        "You review a draft Plan for testability. Can each Behavior be proven by one automated test with no network and no "
        "human? Name Behaviors that cannot, and assumptions whose probes would not work. Reply with JSON "
        "{\"points\": [...]}, at most 3 short points."
    ),
    "risk": (
        "You review a draft Plan for risk. What could go wrong or surprise the builder: unknown code, edge cases, wrong "
        "assumptions? Reply with JSON {\"points\": [...]}, at most 3 short points."
    ),
    "spacing": (
        "You review the order and spacing of a draft Plan's Behaviors. For EACH Behavior, in order, ask: if the code written "
        "for the earlier Behaviors were in place, would this Behavior's test already pass? Name every Behavior where it "
        "would (by number), and every Behavior that hides two steps. A Behavior must need new code beyond the earlier ones. "
        "Reply with JSON {\"points\": [...]}, at most 3 short points; an empty list means the spacing is fine."
    ),
}


def critique_system(angle: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and angle in overrides:
        return overrides[angle]
    return CRITIQUE_SYSTEMS.get(angle) or (
        f"You review a draft Plan from one angle: {angle}. Name concrete problems from that angle only. "
        'Reply with JSON {"points": [...]}, at most 3 short points.'
    )


def critique(angle: str, goal: str, plan_text: str) -> str:
    return f"Review angle: {angle}\nGoal: {goal}\n\nDraft Plan:\n{plan_text}"


SYNTHESIZE_SYSTEM = (
    "You read critiques of a draft Plan and ask the human only what the human alone can answer: preferences, intent, "
    'scope calls. Do not ask what the code could answer. Reply with JSON {"questions": [...]}, at most 5.'
)


def synthesize(goal: str, plan_text: str, critiques: list[tuple[str, str]]) -> str:
    listed = "\n".join(f"- [{angle}] {point}" for angle, point in critiques) or "(no points)"
    return f"Goal: {goal}\n\nDraft Plan:\n{plan_text}\n\nCritiques:\n{listed}"
