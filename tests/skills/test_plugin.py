"""F14 gates on the Skills plugin: what the request carries, cache stability, the file-read tool, and context accounting.
The Messages API is faked (``tests.anthropic_fakes``); nothing here touches the network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import Agent

from harness.agent import AnthropicChatModel, build_api_agent, ollama_view
from harness.context.accounting import Category, account
from harness.skills.catalog import SkillRoot, discover
from harness.skills.plugin import SkillsPlugin
from tests.anthropic_fakes import MODEL, FakeApi, text, tool_use
from tests.skills.helpers import write_skill

BODY = "BODY-MARKER: first list the katas, then compare the labels."
BASE = "You are braid."


def agent_on(api: FakeApi, root: Path, **plugin: Any) -> tuple[Agent, SkillsPlugin]:
    catalog = discover([SkillRoot(root, "project")])
    skills = SkillsPlugin(catalog, **plugin)
    return build_api_agent(AnthropicChatModel(api.client(), MODEL), BASE, plugins=[skills]), skills


def system_of(body: dict[str, Any]) -> str:
    return "\n".join(b["text"] for b in body["system"])


async def test_the_request_carries_each_name_and_description_but_no_body_or_path_and_the_block_is_cache_stable(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "kata-report", "Compare Bench ledger labels.", BODY, files={"scripts/report.py": "print(1)"})
    write_skill(root, "pdf-notes", "Summarise a PDF into notes.", "other body")
    api = FakeApi().reply(text("one")).reply(text("two"))
    agent, _ = agent_on(api, root)
    await agent.invoke_async("hi")
    await agent.invoke_async("again")

    first, second = (system_of(b) for b in api.bodies)
    assert first.startswith(BASE)
    assert "- kata-report: Compare Bench ledger labels." in first and "- pdf-notes: Summarise a PDF into notes." in first
    assert BODY not in first and "other body" not in first and "report.py" not in first
    assert str(tmp_path) not in first and "SKILL.md" not in first  # no absolute path, not even the file name
    assert api.bodies[0]["system"] == api.bodies[1]["system"] and first == second  # byte-identical: the cached prefix holds
    assert api.bodies[1]["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_a_skill_added_between_turns_changes_the_block_once(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "first-skill", "one")
    api = FakeApi().reply(text("a")).reply(text("b")).reply(text("c"))
    agent, _ = agent_on(api, root)
    await agent.invoke_async("1")
    write_skill(root, "second-skill", "two")
    await agent.invoke_async("2")
    await agent.invoke_async("3")
    s1, s2, s3 = (system_of(b) for b in api.bodies)
    assert "second-skill" not in s1 and "- second-skill: two" in s2 and s2 == s3
    assert s2.count("<available_skills>") == 1 and s2.startswith(BASE)  # the old block was replaced, not stacked


async def test_the_body_loads_only_when_the_skill_tool_is_called(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "kata-report", "Compare Bench ledger labels.", BODY, frontmatter="allowed-tools: Bash(*)",
                files={"scripts/report.py": "print(1)", "references/format.md": "FORMAT-TEXT"})
    api = (FakeApi()
           .reply(tool_use("skill", {"name": "kata-report"}), stop_reason="tool_use")
           .reply(tool_use("read_skill_file", {"name": "kata-report", "path": "references/format.md"}, id="toolu_2"), stop_reason="tool_use")
           .reply(text("done")))
    agent, skills = agent_on(api, root)
    await agent.invoke_async("make the report")

    loaded = api.bodies[1]["messages"][-1]["content"][0]
    assert loaded["type"] == "tool_result" and BODY in loaded["content"][0]["text"]
    assert "scripts/report.py" in loaded["content"][0]["text"] and "references/format.md" in loaded["content"][0]["text"]
    assert "allowed-tools: Bash(*)" in loaded["content"][0]["text"] and "display only" in loaded["content"][0]["text"]
    assert str(tmp_path) not in loaded["content"][0]["text"]  # resources are listed relative to the skill
    assert "FORMAT-TEXT" in api.bodies[2]["messages"][-1]["content"][0]["content"][0]["text"]
    assert BODY not in system_of(api.bodies[2])
    assert agent.state.get("skills")["active"] == ["kata-report"]
    assert skills.catalog.has("kata-report")


async def test_read_skill_file_refuses_a_path_out_of_the_skill_and_a_symlink_out(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = write_skill(root, "reader", "reads", files={"references/ok.md": "fine"})
    (tmp_path / "zshrc").write_text("export SECRET=1")
    (skill / "references" / "link.md").symlink_to(tmp_path / "zshrc")
    api = (FakeApi()
           .reply(tool_use("read_skill_file", {"name": "reader", "path": "../../.zshrc"}), stop_reason="tool_use")
           .reply(tool_use("read_skill_file", {"name": "reader", "path": "references/link.md"}, id="toolu_2"), stop_reason="tool_use")
           .reply(tool_use("skill", {"name": "nobody"}, id="toolu_3"), stop_reason="tool_use")
           .reply(text("done")))
    agent, _ = agent_on(api, root)
    await agent.invoke_async("go")
    results = [b for body in api.bodies[1:] for b in body["messages"][-1]["content"] if b["type"] == "tool_result"]
    assert [r.get("is_error") for r in results] == [True, True, True]
    assert "outside the skill" in results[0]["content"][0]["text"] and "outside the skill" in results[1]["content"][0]["text"]
    assert "export SECRET" not in str(api.bodies)
    assert "no skill named 'nobody'" in results[2]["content"][0]["text"]


async def test_accounting_files_the_index_under_system_and_a_body_under_tool_results(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "kata-report", "Compare Bench ledger labels.", "long body " * 200)
    api = FakeApi().reply(tool_use("skill", {"name": "kata-report"}), stop_reason="tool_use").reply(text("done"))
    agent, _ = agent_on(api, root)
    await agent.invoke_async("make the report")

    before = account(*ollama_view(api.bodies[0])[:1])
    after = account(*ollama_view(api.bodies[1])[:1])
    index_chars = len("- kata-report: Compare Bench ledger labels.")
    assert before.chars[Category.SYSTEM] > index_chars and before.chars[Category.TOOL_RESULTS] == 0
    assert after.chars[Category.SYSTEM] == before.chars[Category.SYSTEM]  # the body did not touch the system block
    assert after.chars[Category.TOOL_RESULTS] > len("long body " * 200)
