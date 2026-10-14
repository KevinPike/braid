"""F15 gates on ``run_skill_script``: always an Approval, only through the Sandbox, a from-scratch environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from strands import Agent

from harness.approval import ALWAYS_GATED, ApprovalGate, ApprovalRequest
from harness.sandbox import SeatbeltSandbox
from harness.skills.catalog import SkillRoot, discover
from harness.skills.scripts import build_run_skill_script
from tests.skills.helpers import FakeSandbox, write_skill
from tests.test_approval import ScriptedModel, approver

CALL = ("run_skill_script", {"name": "kata-report", "script": "scripts/report.py", "args": ["word-wrap", "--top 3"]})


def setup(tmp_path: Path, **frontmatter: str) -> tuple[Path, Path, Any, FakeSandbox]:
    project = tmp_path / "project"
    project.mkdir()
    skills = project / ".braid" / "skills"
    skill_dir = write_skill(
        skills, "kata-report", "Compare Bench ledger labels.", frontmatter=frontmatter.get("frontmatter", ""),
        files={"scripts/report.py": "print('report')", "references/notes.md": "not a script"},
    )
    sandbox = FakeSandbox()
    tool = build_run_skill_script(discover([SkillRoot(skills, "project")]), sandbox, project, timeout_s=7.0)
    return project, skill_dir.resolve(), tool, sandbox


def results(agent: Agent) -> list[Any]:
    return [c["toolResult"] for m in agent.messages for c in m["content"] if "toolResult" in c]


def test_run_skill_script_is_always_gated_whatever_the_configuration_says() -> None:
    assert "run_skill_script" in ALWAYS_GATED
    assert "run_skill_script" in ApprovalGate(approver(True, []), gated=[]).gated


async def test_a_denied_script_never_reaches_the_sandbox(tmp_path: Path) -> None:
    project, _, tool, sandbox = setup(tmp_path)
    asked: list[ApprovalRequest] = []
    agent = Agent(model=ScriptedModel(CALL, "ok"), callback_handler=None, tools=[tool], hooks=[ApprovalGate(approver(False, asked))])
    await agent.invoke_async("go")
    assert sandbox.runs == [] and [r.tool for r in asked] == ["run_skill_script"]
    assert results(agent)[0]["status"] == "error" and "denied" in results(agent)[0]["content"][0]["text"]
    assert "scripts/report.py" in asked[0].describe() and "word-wrap" in asked[0].describe() and "network: off" in asked[0].describe()


async def test_an_approved_script_runs_in_the_sandbox_with_the_skill_readable_no_network_and_no_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    project, skill_dir, tool, sandbox = setup(tmp_path)
    asked: list[ApprovalRequest] = []
    agent = Agent(model=ScriptedModel(CALL, "ok"), callback_handler=None, tools=[tool], hooks=[ApprovalGate(approver(True, asked))])
    await agent.invoke_async("go")

    (run,) = sandbox.runs
    assert len(asked) == 1
    assert run.cwd == project.resolve() and run.timeout_s == 7.0
    assert skill_dir in run.policy.readable and project.resolve() in run.policy.writable and run.policy.network is False
    assert run.command == f"python3 {skill_dir}/scripts/report.py word-wrap '--top 3'"
    assert set(run.env) <= {"PATH", "BRAID_SKILL_DIR"} and "secret" not in str(run.env)
    assert "ran" in results(agent)[0]["content"][0]["text"] and results(agent)[0]["status"] == "success"


async def test_network_is_a_sandbox_escape_the_approval_shows(tmp_path: Path) -> None:
    _, _, tool, sandbox = setup(tmp_path)
    asked: list[ApprovalRequest] = []
    call = ("run_skill_script", {"name": "kata-report", "script": "scripts/report.py", "network": True})
    await Agent(model=ScriptedModel(call, "ok"), callback_handler=None, tools=[tool], hooks=[ApprovalGate(approver(True, asked))]).invoke_async("go")
    assert "network: ALLOWED (a Sandbox escape)" in asked[0].describe() and sandbox.runs[0].policy.network is True


@pytest.mark.parametrize("script", ["references/notes.md", "../../../../etc/passwd", "scripts/missing.py", "SKILL.md"])
async def test_only_files_under_the_skills_scripts_folder_run(script: str, tmp_path: Path) -> None:
    _, _, tool, sandbox = setup(tmp_path)
    call = ("run_skill_script", {"name": "kata-report", "script": script})
    agent = Agent(model=ScriptedModel(call, "ok"), callback_handler=None, tools=[tool], hooks=[ApprovalGate(approver(True, []))])
    await agent.invoke_async("go")
    assert sandbox.runs == [] and results(agent)[0]["status"] == "error"


async def test_allowed_tools_in_the_frontmatter_skips_no_approval(tmp_path: Path) -> None:
    _, _, tool, sandbox = setup(tmp_path, frontmatter="allowed-tools: Bash(*) run_skill_script")
    asked: list[ApprovalRequest] = []
    agent = Agent(model=ScriptedModel(CALL, "ok"), callback_handler=None, tools=[tool], hooks=[ApprovalGate(approver(False, asked))])
    await agent.invoke_async("go")
    assert len(asked) == 1 and sandbox.runs == []  # asked, denied, not run


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
async def test_under_seatbelt_the_script_sees_a_from_scratch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    project = (tmp_path / "project").resolve()
    skills = project / ".braid" / "skills"
    write_skill(skills, "envdump", "Prints its environment.", files={"scripts/env.sh": "env\nls .braid/skills\n"})
    tool = build_run_skill_script(discover([SkillRoot(skills, "project")]), SeatbeltSandbox(), project)
    out = await tool(name="envdump", script="scripts/env.sh")
    assert "exit code 0" in out and "envdump" in out
    assert "API_KEY" not in out and "secret" not in out and "BRAID_SKILL_DIR=" in out
