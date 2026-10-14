"""F15 gates in the TUI and under Compaction: ``/skill-name``, the Approval text, the offloader exemption and the
post-compaction "active" line. The Messages API is faked (``tests.anthropic_fakes``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import Agent

from harness.agent import AnthropicChatModel, build_api_agent
from harness.app import HarnessApp
from harness.chat import ApprovalPrompt
from harness.config import ContextConfig, PathsConfig
from harness.context.accounting import TokenEstimator
from harness.context.compaction import CompactionPolicy, Compactor
from harness.context.ledger import Ledger
from harness.context.strategies import build_offloader
from harness.context.trimlog import TrimLog
from harness.skills.catalog import SkillRoot, discover
from harness.skills.plugin import SKILL_BODY_TOKENS, SKILL_TOOLS, SkillsPlugin
from tests.anthropic_fakes import MODEL, FakeApi, text, tool_use
from tests.skills.helpers import GatedApi, app_config, send, submit, until, write_skill

BODY = "BODY-MARKER: list the katas, then compare the labels."


def notes(app: HarnessApp) -> str:
    return "\n".join(str(n.render()) for n in app.query(".note"))


def last_user_text(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    return content if isinstance(content, str) else "\n".join(b.get("text", "") for b in content)


def project_skills(tmp_path: Path) -> Path:
    return tmp_path / "project" / ".braid" / "skills"


# --- gate 1 -------------------------------------------------------------------------------------------------------


async def test_slash_skill_name_queues_behind_a_running_turn_then_sends_one_user_turn_with_the_body_and_args(tmp_path: Path) -> None:
    skills = project_skills(tmp_path)
    write_skill(skills, "kata-report", "Compare Bench ledger labels.", BODY)
    api = GatedApi()
    api.reply(text("first")).reply(text("report done"))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "hello")
        await until(pilot, lambda: app.generating)
        await submit(app, pilot, "/kata-report word-wrap")
        assert app.queue.pending == ("/kata-report word-wrap",)  # waited: a skill is a user turn, not an immediate command
        api.gate.set()
        await until(pilot, lambda: not app.generating and not app.queue)

    assert len(api.bodies) == 2
    sent = last_user_text(api.bodies[1])
    assert BODY in sent and "Arguments: word-wrap" in sent and "'kata-report'" in sent
    assert "/kata-report" not in sent.replace("'kata-report'", "")  # the model gets the expanded turn, not the slash text


async def test_slash_skill_works_with_the_skill_prefix_and_a_built_in_wins_over_a_skill_of_the_same_name(tmp_path: Path) -> None:
    skills = project_skills(tmp_path)
    write_skill(skills, "model", "A skill that is called model.", "MODEL-SKILL-BODY")
    write_skill(skills, "kata-report", "Compare Bench ledger labels.", BODY)
    api = FakeApi().reply(text("by prefix")).reply(text("by prefix, via the shadowed name"))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "/skill kata-report the-args")
        assert BODY in last_user_text(api.bodies[0]) and "Arguments: the-args" in last_user_text(api.bodies[0])

        await send(app, pilot, "/model")  # the built-in: it prints the current model and sends nothing
        assert len(api.bodies) == 1 and "current model" in notes(app)

        await send(app, pilot, "/skill model")  # the way to reach a skill that a built-in shadows
        assert "MODEL-SKILL-BODY" in last_user_text(api.bodies[1])

        await send(app, pilot, "/skill nothing-here")
        await send(app, pilot, "/nothing-here")
        assert len(api.bodies) == 2
        assert "no skill named 'nothing-here'" in notes(app) and "unknown command: /nothing-here" in notes(app)


async def test_the_skills_command_lists_tier_size_shadowing_and_notes_without_sending_anything(tmp_path: Path) -> None:
    skills = project_skills(tmp_path)
    write_skill(skills, "kata-report", "Compare labels.", BODY, frontmatter="allowed-tools: Bash(*)")
    (skills / "broken").mkdir()
    (skills / "broken" / "SKILL.md").write_text("no frontmatter")
    app = HarnessApp(app_config(tmp_path), anthropic_client=FakeApi().client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "/skills")
        shown = notes(app)
        assert "kata-report  [project]" in shown and "tokens" in shown and "allowed-tools: Bash(*) (display only)" in shown
        assert "broken" in shown and "frontmatter" in shown


# --- gate 3 -------------------------------------------------------------------------------------------------------


async def test_the_approval_for_a_script_shows_the_skill_and_what_it_declares_and_n_runs_nothing(tmp_path: Path) -> None:
    skills = project_skills(tmp_path)
    write_skill(skills, "kata-report", "Compare labels.", BODY, frontmatter="allowed-tools: Bash(*)",
                files={"scripts/report.py": "open('ran.txt', 'w').write('x')"})
    call = tool_use("run_skill_script", {"name": "kata-report", "script": "scripts/report.py"})
    api = FakeApi().reply(call, stop_reason="tool_use").reply(text("denied, ok"))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "run the report")
        await until(pilot, lambda: bool(app.query(ApprovalPrompt)))
        shown = str(app.query_one(ApprovalPrompt).render())
        assert "Approve run_skill_script?" in shown and "scripts/report.py" in shown and "network: off" in shown
        assert "allowed-tools: Bash(*)" in shown and "skips no Approval" in shown and "skill source: project" in shown
        await pilot.press("n")
        await until(pilot, lambda: not app.generating)
    assert not (tmp_path / "project" / "ran.txt").exists()
    assert "denied by the user" in str(api.bodies[1]["messages"][-1]["content"])


# --- gate 4 -------------------------------------------------------------------------------------------------------


def model_agent(api: FakeApi, root: Path, tmp_path: Path, *, keep: bool = True, **plugin: Any) -> tuple[Agent, TrimLog]:
    trimlog = TrimLog()
    est = TokenEstimator()
    offloader = build_offloader(
        ContextConfig(offload_tokens=4000), PathsConfig(data_dir=tmp_path / "data"), trimlog, est,
        keep=SKILL_TOOLS if keep else (), keep_tokens=SKILL_BODY_TOKENS,
    )
    assert offloader is not None
    skills = SkillsPlugin(discover([SkillRoot(root, "project")]), trimlog=trimlog, estimator=est, **plugin)
    return build_api_agent(AnthropicChatModel(api.client(), MODEL), "You are braid.", plugins=[offloader, skills]), trimlog


def result_text(body: dict[str, Any]) -> str:
    return str(body["messages"][-1]["content"][0]["content"][0]["text"])


async def test_a_4500_token_body_is_not_offloaded_but_a_larger_one_and_other_tools_still_are(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    medium = "word " * 3600  # ~4500 tokens on the estimator Strands falls back to (4 characters a token)
    write_skill(root, "kata-report", "Compare labels.", BODY + medium)
    write_skill(root, "huge", "Too big.", BODY + "word " * 5400)  # ~6750 tokens
    for name, offloaded in (("kata-report", False), ("huge", True)):
        api = FakeApi().reply(tool_use("skill", {"name": name}), stop_reason="tool_use").reply(text("ok"))
        agent, _ = model_agent(api, root, tmp_path)
        await agent.invoke_async("go")
        assert ("Offloaded" in result_text(api.bodies[1])) is offloaded, name
    medium_tokens = await AnthropicChatModel(FakeApi().client(), MODEL).count_tokens([{"role": "user", "content": [{"text": BODY + medium}]}])
    assert 4000 < medium_tokens <= SKILL_BODY_TOKENS  # the premise: bigger than offload_tokens, under the exemption

    api = FakeApi().reply(tool_use("skill", {"name": "kata-report"}), stop_reason="tool_use").reply(text("ok"))
    agent, _ = model_agent(api, root, tmp_path, keep=False)
    await agent.invoke_async("go")
    assert "Offloaded" in result_text(api.bodies[1])  # the control: without the exemption the same body goes to disk


async def test_compaction_that_drops_an_active_body_adds_the_active_line_and_a_trim_log_row(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "kata-report", "Compare labels.", BODY)
    api = (FakeApi()
           .reply(tool_use("skill", {"name": "kata-report"}), stop_reason="tool_use").reply(text("report done"))
           .reply(text("another answer")).reply(text("a third answer"))
           .reply(tool_use("skill", {"name": "kata-report"}), stop_reason="tool_use").reply(text("reloaded"))
           .reply(text("and on")))
    agent, trimlog = model_agent(api, root, tmp_path)
    await agent.invoke_async("make the report")
    system_before = api.bodies[1]["system"]
    assert "active:" not in str(system_before)  # the body is in history, nothing to say

    async def summarize(_messages: Any) -> str:
        return "- the user asked for a report"

    compactor = Compactor(
        trimlog=trimlog, ledger=Ledger(lambda _m: 100_000), summarize=summarize, num_ctx=lambda: 100_000,
        policy=CompactionPolicy(preserve_recent=1),
    )
    entry = await compactor.compact(agent.messages, "test")
    assert entry is not None and BODY not in str(agent.messages)

    await agent.invoke_async("and now?")
    system_after = "\n".join(b["text"] for b in api.bodies[2]["system"])
    assert "active: kata-report (reload with `skill`)" in system_after
    assert system_after.startswith("You are braid.") and system_after.count("<available_skills>") == 1
    rows = [e for e in trimlog.entries() if e.strategy == "skill"]
    assert len(rows) == 1 and "kata-report" in rows[0].reason and rows[0].tokens_before > 0 and rows[0].tokens_after == 0

    await agent.invoke_async("once more")  # still dropped: the line stays, but the Trim log row is written once
    assert "active: kata-report" in str(api.bodies[3]["system"])
    assert len([e for e in trimlog.entries() if e.strategy == "skill"]) == 1

    await agent.invoke_async("reload it")  # the model calls `skill` again; the next turn's block drops the line
    assert "active: kata-report" in str(api.bodies[4]["system"])
    await agent.invoke_async("carry on")
    assert "active:" not in str(api.bodies[6]["system"])
