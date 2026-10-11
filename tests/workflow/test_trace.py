"""F9 gates: Route records on every conditional edge, and thinking summaries in the run log and both inspectors.

No live calls: the Messages API is faked with ``httpx2.MockTransport`` and the models are scripted."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel
from textual.app import App
from textual.widgets import OptionList, Static

from harness.agent import AnthropicChatModel, build_api_agent, stream_reply
from harness.app import HarnessApp
from harness.config import AnthropicConfig, HarnessConfig, ThinkingDisplay, WorkflowConfig
from harness.inspector import Inspector, RunInspector, call_text, describe_node, entry_label
from harness.sandbox import LocalSandbox, Workspace
from harness.session.store import SessionStore
from harness.usage import UsageMeter
from harness.workflow.backend import Choice, make_model
from harness.workflow.execute import ExecState, build_execution_graph, drive_graph, initial_state
from harness.workflow.model import AnthropicWorkflowModel
from harness.workflow.plan import Plan
from harness.workflow.records import Recorder, Route, RunLog, read_log, route_counts, write_summary
from tests.anthropic_fakes import MODEL, FakeApi, text, thinking
from tests.test_app_anthropic import config as chat_config
from tests.test_app_anthropic import send, until
from tests.workflow.test_anthropic_model import GOOD, Answer, message
from tests.workflow.test_api_runs import CostlyModel
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, NEG, NEG_TEST, NEG_WRONG, ScriptedModel, edit
from tests.workflow.test_plan_loop import run_plan, scripted

SAMPLING = {"temperature", "top_p", "top_k"}
SUMMARY = "The test wants mul(2, 3) == 6, so a product is enough."
NO_ROUTE_AFTER = {"finish", "spend_cap", "critique", "finalize"}  # a fixed edge to END, or (critique) to synthesize


def workflow_model(replies: list[dict[str, Any]], **kw: Any) -> tuple[AnthropicWorkflowModel, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=replies[len(bodies) - 1], headers={"request-id": f"req_{len(bodies)}"})

    client = anthropic.AsyncAnthropic(
        api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0,
    )
    return AnthropicWorkflowModel(client, MODEL, **kw), bodies


def thought(reply: BaseModel | str, summary: str = "") -> dict[str, Any]:
    """A Workflow reply whose content starts with a thinking block, summarized or (empty) omitted."""
    body = message(reply if isinstance(reply, str) else reply.model_dump_json())
    body["content"].insert(0, {"type": "thinking", "thinking": summary, "signature": "sig"})
    return body


# --- gates 1 and 4: the request body --------------------------------------------------------------------------------

@pytest.mark.parametrize("display", ["summarized", "omitted"])
async def test_the_thinking_display_reaches_both_request_bodies_and_no_sampling_parameter_does(
    display: ThinkingDisplay, tmp_path: Path,
) -> None:
    model, bodies = workflow_model([message(GOOD)], thinking_display=display)
    await model.complete("sys", "go", Answer)
    api = FakeApi().reply(text("hi"))
    chat = AnthropicChatModel(api.client(), MODEL, thinking_display=display)
    [d async for d in stream_reply(build_api_agent(chat, "sys"), "hello")]
    for body in (bodies[0], api.bodies[0]):
        assert not SAMPLING & body.keys()
        if display == "summarized":
            assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        else:
            assert "display" not in body.get("thinking", {})


def test_the_display_defaults_to_summarized_and_the_backend_passes_it_on() -> None:
    assert AnthropicConfig().thinking_display == "summarized"
    config = HarnessConfig(anthropic=AnthropicConfig(thinking_display="omitted"))
    model = make_model(config, Choice("anthropic", MODEL), None)
    assert isinstance(model, AnthropicWorkflowModel) and model.thinking is None
    assert make_model(HarnessConfig(), Choice("anthropic", MODEL), None).thinking == {  # type: ignore[union-attr]
        "type": "adaptive", "display": "summarized",
    }


# --- gate 2: thinking summaries in the run log and the inspectors ---------------------------------------------------

async def run_exec(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, model: Any, answers: list[str], *,
    behaviors: list[str] | None = None, settings: WorkflowConfig | None = None,
) -> tuple[ExecState, Recorder, RunLog]:
    ws = await Workspace.create(fixture_repo, tmp_path / "workspaces", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    plan = Plan(goal="arithmetic helpers", behaviors=behaviors or ["mul multiplies"], files=["src/mod.py"])
    rec, log = Recorder(), RunLog(tmp_path / "runs" / "run1.jsonl")
    graph = build_execution_graph(model, LocalSandbox(), ws, settings or WorkflowConfig(), recorder=rec, log=log)
    final = await drive_graph(graph, initial_state(plan), lambda _p: answers.pop(0))
    return final, rec, log


async def test_a_thinking_summary_reaches_the_run_log_and_the_run_inspector(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model, _ = workflow_model([thought(MUL_TEST, SUMMARY), thought(MUL_IMPL), thought(edit("src/mod.py", MUL))])
    final, _rec, log = await run_exec(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"])
    assert final["status"] == "kept"
    lines = log.lines()
    write_test = next(r for r in lines if r["node"] == "write_test")
    write_impl = next(r for r in lines if r["node"] == "write_impl")
    assert write_test["thinking"] == SUMMARY and "thinking" not in write_impl  # an omitted (empty) summary adds nothing
    shown = describe_node(write_test)
    assert shown.index("--- reply ---") < shown.index("--- thinking summary ---\n" + SUMMARY)

    class Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(RunInspector("run1", read_log(log.path)))

    app = Host()
    async with app.run_test() as pilot:
        options = app.screen.query_one(OptionList)
        labels = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert labels[0].endswith("check_assumptions") and "→ write_test  code: open_questions=[]" in labels[1]
        index = next(i for i, r in enumerate(lines) if r["node"] == "write_test")
        for _ in range(index):
            await pilot.press("down")
        await pilot.pause(0.1)
        assert SUMMARY in str(app.screen.query_one("#run-detail", Static).render())


async def test_the_chat_inspector_shows_the_replys_thinking_summary(tmp_path: Path) -> None:
    api = FakeApi().reply(thinking("sig-1", "The user greets me; greet back."), text("Hello there"))
    app = HarnessApp(chat_config(tmp_path), anthropic_client=api.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "hi")
        [call] = app.store.calls(app.session_id)
        assert call.thinking == "The user greets me; greet back."
        app.action_inspect()
        await until(pilot, lambda: isinstance(app.screen, Inspector))
        await pilot.pause(0.1)
        shown = str(app.screen.query_one("#inspector-payload", Static).render())
        assert shown.startswith("--- thinking summary ---\nThe user greets me; greet back.") and '"model"' in shown


def test_a_store_from_before_f9_gains_the_thinking_column(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE calls (session_id TEXT NOT NULL, n INTEGER NOT NULL, at TEXT NOT NULL, model TEXT NOT NULL,"
               " num_ctx INTEGER NOT NULL, payload TEXT NOT NULL, estimate INTEGER, prompt_eval_count INTEGER,"
               " eval_count INTEGER, PRIMARY KEY (session_id, n))")
    db.commit()
    db.close()
    store = SessionStore(path)
    n = store.record_call("s1", MODEL, 0, {"model": MODEL})
    store.complete_call("s1", n, 10, 2, "short thought")
    [call] = store.calls("s1")
    assert call.thinking == "short thought" and call_text(call).startswith("--- thinking summary ---\nshort thought")
    assert call_text(replace(call, thinking="")) == json.dumps({"model": MODEL}, indent=2)  # Ollama, or omitted


async def test_a_chat_reply_with_omitted_thinking_records_no_summary() -> None:
    api = FakeApi().reply(thinking(), text("hi"))
    meter = UsageMeter(MODEL, 1_000_000)
    chat = AnthropicChatModel(api.client(), MODEL, meter=meter, thinking_display="omitted")
    [d async for d in stream_reply(build_api_agent(chat, "sys"), "hello")]
    assert [c.thinking for c in meter.calls] == [""]


# --- gate 3: one Route record per conditional edge, counted in the summary ------------------------------------------

def check_routes(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every node record with a conditional edge out is followed by exactly one Route record, which names that node and
    the node that runs next. Returns the Route records."""
    for i, r in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if r["node"] == "route":
            assert lines[i - 1]["node"] == r["from"] != "route"
            assert r["basis"] in ("code", "model", "human")
            if r["to"] != "__end__":
                assert nxt is not None and nxt["node"] == r["to"]
            else:
                assert nxt is None
        elif r["node"] not in NO_ROUTE_AFTER:
            assert nxt is not None and nxt["node"] == "route" and nxt["from"] == r["node"], f"no Route record after {r['node']}"
    return [r for r in lines if r["node"] == "route"]


async def test_every_conditional_edge_of_a_scripted_build_writes_one_route_record_and_the_summary_counts_them(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    polished = edit("src/mod.py", '"""Helpers."""\n\n' + NEG)
    model = ScriptedModel([MUL_TEST, MUL_IMPL, NEG_TEST, edit("src/mod.py", NEG_WRONG), edit("src/mod.py", NEG), polished])
    final, rec, log = await run_exec(
        fixture_repo, fixture_venv, tmp_path, model, ["yes", "yes"], behaviors=["mul multiplies", "neg negates"],
    )
    assert final["status"] == "applied"
    lines = log.lines()
    routes = check_routes(lines)
    assert len(routes) == len([r for r in lines if r["node"] != "route"]) - 1  # every node but finish
    by_edge = {(r["from"], r["to"]): r for r in routes}
    assert by_edge[("run_red", "write_impl")]["field"] == "red_verdict" and by_edge[("run_red", "write_impl")]["value"] == "red"
    assert by_edge[("run_green", "write_impl")]["field"] == "attempts" and by_edge[("run_green", "write_impl")]["value"] == 1
    assert by_edge[("next_behavior", "sign_off")]["value"] == 2
    sign_off = by_edge[("sign_off", "polish")]
    assert (sign_off["basis"], sign_off["field"], sign_off["value"]) == ("human", "answer", "yes")  # an interrupt decided
    assert by_edge[("write_test", "run_red")]["field"] is None  # a fixed edge: only the Spend cap could divert it
    counts = route_counts(lines)
    assert counts == rec.routes == {"code": len(routes) - 1, "model": 0, "human": 1}
    summary = write_summary(
        tmp_path / "summaries", model="fake", project=fixture_repo, no_hints=False, status=final.get("status"), planned=2,
        rec=rec, workspace=tmp_path, run_log=log.path,
    )
    assert json.loads(summary.read_text())["routes"] == counts


async def test_the_spend_cap_route_names_the_cost_that_decided_it(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = CostlyModel([MUL_TEST, MUL_IMPL], price=0.06)
    final, rec, log = await run_exec(fixture_repo, fixture_venv, tmp_path, model, [])
    assert final["status"] == "spend_cap"
    last = check_routes(log.lines())[-1]
    assert (last["from"], last["to"], last["field"], last["value"]) == ("write_impl", "spend_cap", "cost_usd", pytest.approx(0.12))
    assert rec.routes["code"] == len(check_routes(log.lines()))


async def test_a_no_hints_run_routes_on_code_where_a_human_would_have_answered(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    ws = await Workspace.create(fixture_repo, tmp_path / "workspaces", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    rec, log = Recorder(), RunLog(tmp_path / "runs" / "run1.jsonl")
    plan = Plan(goal="g", behaviors=["mul multiplies"], files=["src/mod.py"])
    graph = build_execution_graph(
        ScriptedModel([MUL_TEST, MUL_IMPL]), LocalSandbox(), ws, WorkflowConfig(max_polish_rounds=0), hints=False,
        recorder=rec, log=log,
    )
    await drive_graph(graph, initial_state(plan), lambda _p: "no")
    sign_off = next(r for r in check_routes(log.lines()) if r["from"] == "sign_off")
    assert (sign_off["to"], sign_off["basis"], sign_off["value"]) == ("finish", "code", "yes")
    assert rec.routes["human"] == 0


async def test_every_conditional_edge_of_the_plan_loop_writes_one_route_record(fixture_repo: Path, tmp_path: Path) -> None:
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["research: how are tests laid out?", "drop the zero case", "approve"])
    lines = run.log.lines()
    routes = check_routes(lines)
    fan_outs = [r for r in routes if r["from"] == "draft"]
    assert len(fan_outs) == 3 and all(r["to"] == "critique" and r["field"] == "critique_angles" for r in fan_outs)
    reviews = [r for r in routes if r["from"] == "human_review"]
    assert [(r["to"], r["basis"], r["field"]) for r in reviews] == [
        ("research", "human", "answer"), ("draft", "human", "answer"), ("finalize", "human", "answer"),
    ]
    assert route_counts(lines) == run.rec.routes == {"code": len(routes) - 3, "model": 0, "human": 3}


async def test_the_round_limit_route_has_a_code_basis(fixture_repo: Path, tmp_path: Path) -> None:
    run = await run_plan(fixture_repo, tmp_path, scripted(), ["more tests", "more", "more"], WorkflowConfig(max_rounds=3))
    last = check_routes(run.log.lines())[-1]
    assert (last["from"], last["to"], last["basis"], last["field"], last["value"]) == ("human_review", "finalize", "code", "round", 3)


# --- the hook F10 uses: a model-basis Route record ------------------------------------------------------------------

def test_a_model_basis_route_record_carries_the_model_request_id_reason_and_thinking(tmp_path: Path) -> None:
    route = Route.by_model(
        "write_test", "verdict", "test_wrong", model="claude-sonnet-5-5", request_id="req_42",
        reason="the goal says a strike scores 10 plus the next two rolls", thinking="The test expects 10; the goal says more.",
    )
    log, rec = RunLog(tmp_path / "run.jsonl"), Recorder()
    rec.add_route(route)
    log.route("arbitrate", route, behavior=1)
    [line] = log.lines()
    assert {k: line[k] for k in ("node", "from", "to", "basis", "field", "value", "model", "request_id", "behavior")} == {
        "node": "route", "from": "arbitrate", "to": "write_test", "basis": "model", "field": "verdict",
        "value": "test_wrong", "model": "claude-sonnet-5-5", "request_id": "req_42", "behavior": 1,
    }
    assert line["reason"].startswith("the goal says") and line["thinking"].startswith("The test expects 10")
    assert route_counts([line]) == rec.routes == {"code": 0, "model": 1, "human": 0}
    shown = describe_node(line)
    assert "→ write_test  model: verdict=\"test_wrong\"" in shown and "req_42" in shown and "--- thinking summary ---" in shown
    assert entry_label(0, line) == '  1   → write_test  model: verdict="test_wrong"'


def test_a_code_route_record_has_no_claim_and_a_long_value_is_cut() -> None:
    record = Route("write_impl", "answer", "x" * 500).record("stuck")
    assert set(record) == {"from", "to", "basis", "field", "value"} and len(record["value"]) == 201


def test_a_checkpoint_from_before_f9_loads_with_no_routes() -> None:
    old = Recorder().dump()
    del old["routes"]
    rec = Recorder()
    rec.load(old)
    assert rec.routes == {"code": 0, "model": 0, "human": 0}
