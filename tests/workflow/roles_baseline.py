"""F11 gate 4's scripted run: every (system, prompt) the Execution loop sends, with the run's paths and timings masked.

``roles_baseline_prompts.json`` holds what this run sent at 15b02c0, before Roles; ``test_roles.py`` runs it again with no
Roles configured and compares byte for byte. To capture it again (only if a prompt is meant to change)::

    uv run python -m tests.workflow.roles_baseline
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from harness.config import WorkflowConfig
from harness.sandbox import LocalSandbox, RunResult, SandboxPolicy
from harness.workflow.execute import Ruling
from harness.workflow.invariants import Invariant
from tests.workflow.test_execute import MUL_IMPL, MUL_TEST, NEG, NEG_TEST, ScriptedModel, dispute, edit, run_loop

T = TypeVar("T", bound=BaseModel)

BASELINE = Path(__file__).parent / "data" / "roles_baseline_prompts.json"
MUL_RULE = Invariant(text="mul(a, b) is a times b", fn="mul", check="result == args[0] * args[1]")
RIGHT = Ruling(verdict="test_right", reason="The goal says mul multiplies; 6 is right.")


@dataclass
class Recording(ScriptedModel):
    """A ``ScriptedModel`` that keeps each call's system prompt too."""

    sent: list[tuple[str, str]] = field(default_factory=list)

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        self.sent.append((system, prompt))
        return await super().complete(system, prompt, schema)


class Steady(LocalSandbox):
    """``LocalSandbox`` with pytest's output cut to what does not depend on the pytest version: the ``E`` lines and the
    result lines. The harness builds its prompts from this output either way; the comparison is of what it does with it."""

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = 120.0,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        result = await super().run(command, cwd=cwd, policy=policy, timeout_s=timeout_s, env=env)
        if "pytest" not in command:
            return result
        kept = [line for line in result.output.splitlines()
                if re.match(r"(E |FAILED|ERROR|PASSED)", line) or re.search(r"\b(passed|failed|errors?) in ", line)]
        return replace(result, output="\n".join(kept) + "\n")


def mask(text: str, roots: list[Path]) -> str:
    for root in roots:
        text = text.replace(str(root.resolve()), "<tmp>").replace(str(root), "<tmp>")
    return re.sub(r"\d+\.\d+s\b", "<t>s", text)


async def scripted_prompts(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, settings: WorkflowConfig | None = None,
) -> list[dict[str, str]]:
    """Two Behaviors with an Invariant: a test rejected once (a syntax error), a Dispute the Arbiter rules on, an
    implementation, Sign-off, Polish. Returns ``[{"role", "system", "prompt"}]`` in call order."""
    broken = edit("tests/test_behavior_1.py", "def test_x(:\n    pass\n")
    model = Recording([
        broken, MUL_TEST, dispute("test_b1", "I believe mul(2, 3) is 5"), MUL_IMPL, NEG_TEST, edit("src/mod.py", NEG),
        edit("src/mod.py", '"""Helpers."""\n\n' + NEG),
    ])
    arbiter = Recording([RIGHT])
    await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], assumptions=[], invariants=[MUL_RULE],
                   arbiter=arbiter, settings=settings or WorkflowConfig(arbiter="claude-sonnet-5-5"), sandbox=Steady())
    roots = [tmp_path, fixture_repo]
    sent = [("model", s, p) for s, p in model.sent] + [("arbiter", s, p) for s, p in arbiter.sent]
    return [{"role": who, "system": mask(s, roots), "prompt": mask(p, roots)} for who, s, p in sent]


async def _capture() -> None:
    from tests.conftest import make_fixture_repo

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        venv = root / "venv" / ".venv"
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True, capture_output=True)
        subprocess.run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "pytest"], check=True,
                       capture_output=True)
        sent = await scripted_prompts(make_fixture_repo(root / "project"), venv, root / "t")
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(sent, indent=1) + "\n")
    print(f"{len(sent)} prompts -> {BASELINE}")


if __name__ == "__main__":
    asyncio.run(_capture())
