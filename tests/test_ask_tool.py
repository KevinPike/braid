"""C3 gates: the chat tool ``ask_user_question`` over the fake Anthropic API, the Approval and Choice sharing the human,
Ctrl-C leaving no unpaired tool use, and ``--demo choice`` booting offline (ADR 0010)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.chat import ApprovalPrompt, ChoicePrompt, Reply
from harness.choice import tool_schema
from harness.config import AskConfig, load_config
from harness.demo import demo_app
from harness.tools.ask import DESCRIPTION, TOOL_NAME
from tests.anthropic_fakes import FakeApi, text, tool_use
from tests.test_app_anthropic import config, until

STORAGE: dict[str, Any] = {
    "header": "Storage", "question": "Where should the kata keep its users?",
    "options": [{"label": "SQLite", "description": "one file"}, {"label": "JSON file"}, {"label": "In memory"}]}
ASK: dict[str, Any] = {"questions": [STORAGE]}


def ask(id: str = "toolu_1", args: dict[str, Any] | None = None) -> dict[str, Any]:
    return tool_use(TOOL_NAME, ASK if args is None else args, id=id)


async def submit(app: HarnessApp, pilot: Pilot[None], prompt: str) -> None:
    app.query_one("#prompt", Input).value = prompt
    await pilot.press("enter")
    await pilot.pause(0.05)


def tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for m in body["messages"] for b in m["content"] if isinstance(b, dict) and b["type"] == "tool_result"]


def records(app: HarnessApp) -> list[str]:
    return [str(n.render()) for n in app.query(Static) if n.has_class("record")]


async def test_a_scripted_tool_use_mounts_the_widget_and_the_answer_returns_as_the_tool_result(tmp_path: Path) -> None:
    api = FakeApi().reply(ask(), stop_reason="tool_use").reply(text("SQLite it is."))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "which store?")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.pause(0.05)
        assert app.query_one(ChoicePrompt).border_title == "Choice from claude-haiku-5-5"
        await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating and len(api.bodies) == 2)
        [result] = tool_results(api.bodies[1])
        assert result["tool_use_id"] == "toolu_1" and not result.get("is_error")
        assert json.loads(result["content"][0]["text"]) == {"answers": {STORAGE["question"]: "SQLite"}}
        assert [r.text for r in app.query(Reply)][-1].strip().endswith("SQLite it is.")
        assert records(app)[0].startswith("✓ Storage? SQLite (")
        [row] = app.choices.entries()
        assert (row.source, row.model, row.result.status) == ("chat", "claude-haiku-5-5", "answered")


async def test_the_request_carries_the_pydantic_schema_as_the_tool(tmp_path: Path) -> None:
    api = FakeApi().reply(text("hi"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "hello")
        await until(pilot, lambda: not app.generating)
    [tool] = [t for t in api.bodies[0]["tools"] if t["name"] == TOOL_NAME]
    assert tool == {"name": TOOL_NAME, "description": DESCRIPTION, "input_schema": tool_schema()}
    assert "multiSelect" in json.dumps(tool["input_schema"]) and "$defs" in tool["input_schema"]


async def test_the_tool_is_off_when_the_config_says_so(tmp_path: Path) -> None:
    api = FakeApi().reply(text("hi"))
    cfg = config(tmp_path)
    cfg.tools.ask = False
    app = HarnessApp(cfg, anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "hello")
        await until(pilot, lambda: not app.generating)
    assert TOOL_NAME not in [t["name"] for t in api.bodies[0]["tools"]]


async def test_the_fourth_call_in_a_turn_is_refused_and_never_shown(tmp_path: Path) -> None:
    api = FakeApi()
    for n in range(1, 5):
        api.reply(ask(f"toolu_{n}"), stop_reason="tool_use")
    api.reply(text("deciding myself"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "ask a lot")
        shown: list[ChoicePrompt] = []
        for _ in range(3):
            await until(pilot, lambda: bool(app.query(ChoicePrompt)) and app.query_one(ChoicePrompt) not in shown)
            shown.append(app.query_one(ChoicePrompt))
            await pilot.pause(0.05)
            await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating)
        assert not app.query(ChoicePrompt) and len(api.bodies) == 5
        refused = tool_results(api.bodies[4])[-1]
        assert refused["is_error"] and json.loads(refused["content"][0]["text"]) == {
            "status": "refused", "reason": "ask limit for this turn reached; decide or stop"}
        assert [r.result.status for r in app.choices.entries()] == ["answered"] * 3 + ["refused"]  # the refusal is a row too


async def test_a_confirmation_is_refused_with_the_reason_and_the_model_can_retry(tmp_path: Path) -> None:
    confirm = {"questions": [{**STORAGE, "options": [{"label": "Yes"}, {"label": "No"}]}]}
    api = FakeApi().reply(ask(args=confirm), stop_reason="tool_use").reply(ask("toolu_2"), stop_reason="tool_use").reply(text("ok"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "go")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        first = json.loads(tool_results(api.bodies[1])[0]["content"][0]["text"])
        assert first["status"] == "refused" and "questions[0].options" in first["reason"] and "confirm" in first["reason"]
        await pilot.press("2", "enter")
        await until(pilot, lambda: not app.generating)
        refused, answered = app.choices.entries()
        assert (refused.result.status, answered.result.status) == ("refused", "answered")
        assert "confirm" in refused.result.reason and refused.latency_s == 0.0 and refused.source == "chat"
        assert [o.label for o in refused.request.questions[0].options] == ["Yes", "No"]  # what the model sent, as logged


async def test_two_refusals_stop_the_tool_for_the_turn(tmp_path: Path) -> None:
    bad: dict[str, Any] = {"questions": []}
    api = FakeApi()
    for n in range(1, 4):
        api.reply(ask(f"toolu_{n}", bad), stop_reason="tool_use")
    api.reply(text("giving up"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "go")
        await until(pilot, lambda: not app.generating)
        third = json.loads(tool_results(api.bodies[3])[-1]["content"][0]["text"])
        assert third["status"] == "refused" and "limit" in third["reason"]
        rows = app.choices.entries()  # all three are rows, the invalid ones with a placeholder request
        assert [r.result.status for r in rows] == ["refused"] * 3 and rows[0].request.questions[0].header == "refused"
        assert "limit" in rows[2].result.reason


async def test_ctrl_c_with_the_widget_open_leaves_no_unpaired_tool_use_and_answers_nothing(tmp_path: Path) -> None:
    api = FakeApi().reply(ask(), stop_reason="tool_use").reply(text("never"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "which store?")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.generating and not app.query(ChoicePrompt))
        await pilot.pause(0.2)
        assert len(api.bodies) == 1  # the model never saw a result
        assert app._agent is not None
        assert not any("toolUse" in b for m in app._agent.messages for b in m["content"])
        assert "✗ choice cancelled" in records(app)
        [row] = app.choices.entries()
        assert row.result.status == "cancelled"


async def test_a_gated_shell_and_a_choice_in_one_turn_open_one_after_the_other(tmp_path: Path) -> None:
    api = FakeApi().reply(ask(), tool_use("shell", {"command": "true"}, id="toolu_2"), stop_reason="tool_use").reply(text("done"))
    cfg = config(tmp_path)
    cfg.tools.shell = True
    app = HarnessApp(cfg, anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "both")
        seen: list[str] = []
        while app.generating or not seen:
            choice, approval = bool(app.query(ChoicePrompt)), bool(app.query(ApprovalPrompt))
            assert not (choice and approval), "a Choice and an Approval were open together"
            if choice and "choice" not in seen:
                seen.append("choice")
                await pilot.pause(0.05)
                await pilot.press("1", "enter")
            elif approval and "approval" not in seen:
                seen.append("approval")
                await pilot.pause(0.05)
                await pilot.press("n")
            await pilot.pause(0.02)
        assert sorted(seen) == ["approval", "choice"]
        results = {r["tool_use_id"]: r for r in tool_results(api.bodies[1])}
        assert json.loads(results["toolu_1"]["content"][0]["text"])["answers"]
        assert results["toolu_2"]["is_error"] and "denied" in results["toolu_2"]["content"][0]["text"]


async def test_a_second_concurrent_ask_in_one_message_is_refused(tmp_path: Path) -> None:
    api = FakeApi().reply(ask("toolu_1"), ask("toolu_2"), stop_reason="tool_use").reply(text("ok"))
    app = HarnessApp(config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "ask twice")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.pause(0.05)
        await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating)
        results = {r["tool_use_id"]: json.loads(r["content"][0]["text"]) for r in tool_results(api.bodies[1])}
        assert "answers" in results["toolu_1"] and results["toolu_2"]["status"] == "refused"


def test_ask_defaults() -> None:
    assert (AskConfig().calls_per_turn, AskConfig().refusals_per_turn, AskConfig().timeout_s) == (3, 2, 0.0)


def test_the_ask_table_in_harness_toml_is_parsed(tmp_path: Path) -> None:
    path = tmp_path / "braid.toml"
    path.write_text("[ask]\ncalls_per_turn = 1\nrefusals_per_turn = 5\ntimeout_s = 12.5\n\n[tools]\nask = false\n")
    cfg = load_config(path)
    assert (cfg.ask.calls_per_turn, cfg.ask.refusals_per_turn, cfg.ask.timeout_s, cfg.tools.ask) == (1, 5, 12.5, False)
    path.write_text("[ask]\ntimeout_s = \"soon\"\n")
    with pytest.raises(ValueError, match="timeout_s"):
        load_config(path)


async def test_the_configured_calls_per_turn_is_what_the_tool_enforces(tmp_path: Path) -> None:
    api = FakeApi().reply(ask("toolu_1"), stop_reason="tool_use").reply(ask("toolu_2"), stop_reason="tool_use").reply(text("ok"))
    cfg = config(tmp_path)
    cfg.ask = AskConfig(calls_per_turn=1)
    app = HarnessApp(cfg, anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "ask twice")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.pause(0.05)
        await pilot.press("1", "enter")
        await until(pilot, lambda: not app.generating)
        assert json.loads(tool_results(api.bodies[2])[-1]["content"][0]["text"])["status"] == "refused"


async def test_a_timeout_gives_timed_out_to_the_model_and_the_log_and_chooses_nothing(tmp_path: Path) -> None:
    api = FakeApi().reply(ask(), stop_reason="tool_use").reply(text("I will decide myself."))
    cfg = config(tmp_path)
    cfg.ask = AskConfig(timeout_s=0.2)
    app = HarnessApp(cfg, anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "which store?")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await until(pilot, lambda: not app.generating)
        assert not app.query(ChoicePrompt) and "✗ choice cancelled" not in records(app)
        [result] = tool_results(api.bodies[1])
        assert not result.get("is_error")
        assert json.loads(result["content"][0]["text"]) == {
            "status": "timed_out", "reason": "no answer within 0.2 s; nothing was chosen"}
        [row] = app.choices.entries()
        assert row.result.status == "timed_out" and not row.result.answers and row.latency_s >= 0.2


# --- the offline demo -----------------------------------------------------------------------------------------------


async def test_demo_choice_boots_without_a_key_and_asks_a_choice_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    app = demo_app("choice", data_dir=tmp_path / "data")
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await submit(app, pilot, "start")
        await until(pilot, lambda: bool(app.query(ChoicePrompt)))
        await pilot.pause(0.05)
        await pilot.press("1", "enter")  # Hashing: argon2-cffi
        await pilot.pause(0.05)
        await pilot.press("2", "enter")  # Storage: JSON file
        await until(pilot, lambda: not app.generating)
        said = [r.text for r in app.query(Reply)][-1]
        assert "argon2-cffi" in said and "JSON file" in said and "scripted demo" in said
        assert [r.result.status for r in app.choices.entries()] == ["answered"]
        banner = "\n".join(str(n.render()) for n in app.query(".note"))
        assert "scripted" in banner and "offline" in banner and "data leaves this machine" not in banner
    assert not (tmp_path / ".braid").exists()  # nothing written to the real data dir


def test_an_unknown_demo_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown demo"):
        demo_app("nope", data_dir=tmp_path)
