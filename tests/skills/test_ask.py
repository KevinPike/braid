"""F17 gate 2 and the skill side of C6: a skill that says "ask which kata" makes the model call ``ask_user_question``, the
Choice is asked as ``skill:<name>``, and a skill whose ``allowed-tools`` leaves the tool out is not offered it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from harness.app import HarnessApp
from harness.chat import ChoicePrompt
from harness.tools.ask import TOOL_NAME
from tests.anthropic_fakes import FakeApi, text, tool_use
from tests.skills.helpers import app_config, send, submit, until, write_skill

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "skills" / "kata-pick"
KATA = {"questions": [{
    "header": "Kata", "question": "Which kata do you want to practise?",
    "options": [{"label": "word-wrap", "description": "break text at a width"}, {"label": "fizzbuzz", "description": "the classic"}]}]}


def skills_dir(tmp_path: Path) -> Path:
    return tmp_path / "project" / ".braid" / "skills"


def install_fixture(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, skills_dir(tmp_path) / "kata-pick")


def tool_names(body: dict[str, Any]) -> set[str]:
    return {t["name"] for t in body["tools"]}


def tool_result_texts(body: dict[str, Any]) -> list[str]:
    return [str(c["text"]) for m in body["messages"] for b in m["content"]
            if isinstance(b, dict) and b["type"] == "tool_result" for c in b["content"]]


async def test_a_skill_that_says_ask_which_kata_calls_the_tool_and_the_turn_continues_with_the_answer(tmp_path: Path) -> None:
    install_fixture(tmp_path)
    api = FakeApi().reply(tool_use(TOOL_NAME, KATA), stop_reason="tool_use").reply(text("Starting fizzbuzz."))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "/kata-pick")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        assert "ask_user_question" in api.bodies[0]["messages"][-1]["content"][0]["text"]  # the skill body reached the model
        await pilot.press("2", "enter")
        await until(pilot, lambda: not app.generating and len(api.bodies) == 2)
        [answer] = tool_result_texts(api.bodies[1])
        assert json.loads(answer) == {"answers": {KATA["questions"][0]["question"]: "fizzbuzz"}}
        [row] = app.choices.entries()
        assert (row.source, row.result.status) == ("skill:kata-pick", "answered")


async def test_a_skill_the_model_loads_itself_sets_the_source_for_later_questions(tmp_path: Path) -> None:
    install_fixture(tmp_path)
    api = (FakeApi().reply(tool_use("skill", {"name": "kata-pick"}, id="toolu_1"), stop_reason="tool_use")
           .reply(tool_use(TOOL_NAME, KATA, id="toolu_2"), stop_reason="tool_use").reply(text("done")))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "pick me a kata")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating)
    assert [r.source for r in app.choices.entries()] == ["skill:kata-pick"]


async def test_a_skill_that_lists_allowed_tools_without_the_ask_tool_is_not_offered_it(tmp_path: Path) -> None:
    write_skill(skills_dir(tmp_path), "no-asking", "Never asks.", "Decide by yourself.", frontmatter="allowed-tools: Bash(*) read_file")
    write_skill(skills_dir(tmp_path), "may-ask", "May ask.", "Ask if needed.", frontmatter="allowed-tools: ask_user_question")
    api = FakeApi().reply(text("one")).reply(text("two")).reply(text("three")).reply(text("four"))
    app = HarnessApp(app_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "hello")
        await send(app, pilot, "/no-asking")
        await send(app, pilot, "/may-ask")
        await send(app, pilot, "/no-asking")
        await send(app, pilot, "still here?")
    offered = [TOOL_NAME in tool_names(b) for b in api.bodies]
    assert offered == [True, False, True, False, False]  # a skill with no opinion, or one that names the tool, keeps it
