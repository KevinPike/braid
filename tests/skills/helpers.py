"""Fixture skill trees and a gated fake Anthropic API for the Skills gates (F14, F15)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual.pilot import Pilot
from textual.widgets import Input

from harness.app import HarnessApp
from harness.config import ContextConfig, DecideConfig, HarnessConfig, PathsConfig, SkillsConfig, ToolsConfig
from harness.sandbox import RunResult, SandboxPolicy
from tests.anthropic_fakes import FakeApi


def write_skill(
    root: Path, name: str, description: str = "Does a thing.", body: str = "Step one.\nStep two.", *,
    frontmatter: str = "", files: Mapping[str, str] | None = None,
) -> Path:
    """``<root>/<name>/SKILL.md`` plus any ``files`` (relative path -> text); returns the skill's directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    extra = f"{frontmatter}\n" if frontmatter else ""
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n")
    for rel, text in (files or {}).items():
        (directory / rel).parent.mkdir(parents=True, exist_ok=True)
        (directory / rel).write_text(text)
    return directory


def app_config(tmp_path: Path, *, context: ContextConfig | None = None, **skills: Any) -> HarnessConfig:
    """A config for an app on the fake API, with the project at ``tmp_path/project`` and skills read only from
    ``<project>/.braid/skills``, never from the real home folder."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return HarnessConfig(
        paths=PathsConfig(data_dir=tmp_path / "data"), tools=ToolsConfig(root=project, shell=False, write=False),
        context=context or ContextConfig(preserve_recent=2, offload_tokens=0), decide=DecideConfig(enabled=True),
        skills=SkillsConfig(roots=[".braid/skills"], **skills),
    )


@dataclass
class GatedApi(FakeApi):
    """A fake API whose replies wait for ``gate``: a turn stays running until the test opens it."""

    gate: asyncio.Event = field(default_factory=asyncio.Event)

    async def _body(self, events: list[bytes]) -> AsyncIterator[bytes]:
        await self.gate.wait()
        async for chunk in super()._body(events):
            yield chunk


async def until(pilot: Pilot[None], predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for the condition")
        await pilot.pause(0.02)


async def submit(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause(0.05)


async def send(app: HarnessApp, pilot: Pilot[None], text: str) -> None:
    await submit(app, pilot, text)
    await until(pilot, lambda: not app.generating)


@dataclass
class RecordedRun:
    command: str
    cwd: Path
    policy: SandboxPolicy
    env: Mapping[str, str]
    timeout_s: float


@dataclass
class FakeSandbox:
    """The Sandbox interface, recording every ``run`` and answering with ``output``."""

    output: str = "ran"
    runs: list[RecordedRun] = field(default_factory=list)

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = 120.0, env: Mapping[str, str] | None = None,
    ) -> RunResult:
        self.runs.append(RecordedRun(command, cwd, policy, dict(env or {}), timeout_s))
        return RunResult(0, self.output, False, 0.0)
