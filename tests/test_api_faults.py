"""F8: fault injection on the API path. ``FaultyTransport`` serves 429s, 529s, 500s, dropped connections, cut streams and
bodies that are not JSON past (and within) the SDK's own retries, to Workflows and the chat agent alike. No network."""

from __future__ import annotations

import io
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from harness.anthropic_api import ApiFailure, make_client
from harness.app import HarnessApp
from harness.chat import Reply
from harness.config import AnthropicConfig, HarnessConfig, WorkflowConfig
from harness.sandbox import LocalSandbox, Workspace
from harness.workflow import bench
from harness.workflow.backend import choose, make_model
from harness.workflow.cli import API_ERROR_EXIT, run_build
from harness.workflow.execute import build_execution_graph, drive_graph, initial_state
from harness.workflow.model import AnthropicWorkflowModel, WorkflowModelError
from harness.workflow.plan import Plan
from harness.workflow.records import Recorder, RunLog, read_log
from tests.anthropic_fakes import (
    MODEL, FaultyTransport, InstantRetries, Step, dropped, not_json, overloaded, rate_limited, reply, server_error, streamed, text,
    unauthorized,
)
from tests.test_app_anthropic import config as app_config, no_ollama, send, until  # noqa: F401  (no_ollama is autouse)
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, tree
from tests.workflow.test_plan_loop import run_plan as run_plan_loop, scripted

KEY = "sk-ant-api03-DISTINCTIVE-fake-key-0123456789"
GOOD = '{"value": 3, "why": "because"}'
ONE_CALL = (10 * 0.10 + 5 * 0.50) / 1_000_000  # the usage every healthy ``reply`` reports


class Answer(BaseModel):
    value: int
    why: str


def as_reply(edits: BaseModel) -> Step:
    return reply(edits.model_dump_json())


def workflow_model(script: list[Step], *, max_retries: int = 2) -> tuple[AnthropicWorkflowModel, FaultyTransport]:
    transport = FaultyTransport(script)
    return AnthropicWorkflowModel(transport.client(max_retries=max_retries), MODEL), transport


# --- gate 1: the SDK's retries absorb a transient fault -------------------------------------------------------------

@pytest.mark.parametrize("fault", [rate_limited, overloaded, server_error, dropped], ids=["429", "529", "500", "drop"])
async def test_a_transient_fault_then_success_completes_and_the_request_count_shows_the_retry(fault: Any) -> None:
    model, transport = workflow_model([fault(), reply(GOOD)])
    assert await model.complete("sys", "go", Answer) == Answer(value=3, why="because")
    assert len(transport.bodies) == 2 and transport.bodies[0] == transport.bodies[1]  # the same request, sent again
    assert len(model.calls) == 1 and model.cost_usd == pytest.approx(ONE_CALL)  # the failed request cost nothing


async def test_retry_after_is_honored_and_two_faults_fit_in_the_default_two_retries() -> None:
    model, transport = workflow_model([rate_limited(retry_after="7"), overloaded(), reply(GOOD)])
    await model.complete("sys", "go", Answer)
    assert len(transport.bodies) == 3
    client = model.client
    assert isinstance(client, InstantRetries) and client.max_retries == 2
    waits = client.waits
    assert waits[0] == 7.0 and 0 < waits[1] <= 1.0  # the server's retry-after, then the SDK's own backoff


async def test_a_chat_turn_survives_a_529_within_the_retries(tmp_path: Path) -> None:
    transport = FaultyTransport([overloaded(), streamed(text("Hello there"))])
    app = HarnessApp(app_config(tmp_path), anthropic_client=transport.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "hi")
        assert [r.text for r in app.query(Reply)] == ["Hello there"] and len(transport.bodies) == 2


# --- gate 2: past max_retries, the run ends with api_error, a Checkpoint and nothing applied ---------------------------

@pytest.mark.parametrize(("fault", "kind", "status"), [
    (overloaded, "OverloadedError", 529), (rate_limited, "RateLimitError", 429), (server_error, "InternalServerError", 500),
    (dropped, "APIConnectionError", None),
])
async def test_a_fault_past_max_retries_raises_api_failure_with_its_type_and_request_id(
    fault: Any, kind: str, status: int | None,
) -> None:
    model, transport = workflow_model([fault(), fault()], max_retries=1)
    with pytest.raises(ApiFailure) as caught:
        await model.complete("sys", "go", Answer)
    assert len(transport.bodies) == 2 and model.calls == []  # no usage, no cost
    assert (caught.value.kind, caught.value.status_code) == (kind, status)
    assert caught.value.request_id == (f"req_{2}" if status else None)


@pytest.mark.parametrize("script", [
    [not_json()],  # the SDK does not retry a 200, and json.loads fails
    [reply("x"), server_error(), server_error(), server_error()],  # a failure in the model's own retry: one call cost
], ids=["not-json", "fails-in-the-retry"])
async def test_other_faults_are_api_failures_too_never_invalid_output(script: list[Step]) -> None:
    model, _ = workflow_model(script)
    with pytest.raises(ApiFailure):
        await model.complete("sys", "go", Answer)


async def test_errors_past_max_retries_end_the_run_with_api_error_and_a_resume_finishes_it(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    before = tree(fixture_repo)
    ws = await Workspace.create(fixture_repo, tmp_path / "ws", "run1")
    (ws.path / ".venv").symlink_to(fixture_venv)
    plan = Plan(goal="g", behaviors=["mul multiplies"], files=["src/mod.py"])
    cfg = WorkflowConfig(max_polish_rounds=0)
    saver = InMemorySaver()  # the Checkpoint outlives the first run's graph, model and Recorder
    asked: list[str] = []
    # write_test succeeds; write_impl's request gets three 529s, one more than max_retries=2 allows
    model, transport = workflow_model([as_reply(MUL_TEST), overloaded(), overloaded(), overloaded()])
    rec = Recorder()
    log = RunLog(tmp_path / "runs" / "run1.jsonl")
    graph = build_execution_graph(model, LocalSandbox(), ws, cfg, hints=False, recorder=rec, log=log)

    def ask(payload: dict[str, Any]) -> str:
        asked.append(payload["kind"])
        return "yes"

    final = await drive_graph(graph, initial_state(plan), ask, thread_id="run1", checkpointer=saver)

    assert final["status"] == "api_error" and len(transport.bodies) == 4
    error = final["api_error"]
    assert (error["node"], error["error"], error["status_code"], error["request_id"]) == ("write_impl", "OverloadedError", 529, "req_4")
    assert asked == [] and tree(fixture_repo) == before and ws.path.is_dir()  # nothing applied, the Workspace kept
    last = read_log(log.path)[-1]
    assert (last["node"], last["at"], last["error"], last["request_id"]) == ("api_error", "write_impl", "OverloadedError", "req_4")
    assert rec.cost_usd == pytest.approx(ONE_CALL)  # write_test's call; the failed requests had no usage
    snapshot = await build_execution_graph(model, LocalSandbox(), ws, cfg).compile(checkpointer=saver).aget_state(
        {"configurable": {"thread_id": "run1"}})
    assert snapshot.next == ("write_impl",)  # the Checkpoint before the failed node

    # a resume with a healthy transport: fresh model and Recorder, only the failed node and what follows run
    healthy, again = workflow_model([as_reply(MUL_IMPL)])
    rec2 = Recorder()
    graph = build_execution_graph(healthy, LocalSandbox(), ws, cfg, hints=False, recorder=rec2, log=log)
    final = await drive_graph(
        graph, initial_state(plan), lambda p: "yes", thread_id="run1", checkpointer=saver, resume=True,
        on_resume=lambda values: rec2.load(values["recorder"]),
    )
    assert final["status"] == "applied" and tree(fixture_repo)["src/mod.py"] == MUL
    assert len(again.bodies) == 1  # write_test was not repeated
    assert [b.green for b in rec2.behaviors] == [True] and rec2.behaviors[0].attempts == 1


@pytest.mark.parametrize(("schema", "node"), [("PlanDraft", "draft"), ("CritiqueReply", "critique")])
async def test_the_plan_loop_ends_with_api_error_too_even_in_a_fanned_out_critique(
    fixture_repo: Path, tmp_path: Path, schema: str, node: str,
) -> None:
    def fail(_prompt: str) -> BaseModel:
        raise ApiFailure("OverloadedError", "Overloaded", request_id="req_9", status_code=529)

    replies: dict[str, list[BaseModel | Callable[[str], BaseModel]]] = {schema: [fail]}
    run = await run_plan_loop(fixture_repo, tmp_path, scripted(**replies), [])
    assert run.state["status"] == "api_error" and run.state["api_error"]["node"] == node
    assert run.payloads == [] and not run.plans.exists()  # never reviewed, no Plan saved
    if node == "draft":  # a node the wrapper logs; a Critique branch only stops the run
        last = run.log.lines()[-1]
        assert (last["node"], last["at"], last["request_id"]) == ("api_error", "draft", "req_9")


def cli_config(tmp_path: Path, fixture_venv: Path) -> HarnessConfig:
    return HarnessConfig(workflow=WorkflowConfig(
        workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", plans_dir=tmp_path / "plans", max_polish_rounds=0,
        test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider",
    ))


async def test_the_build_cli_exits_with_api_error_and_the_summary_says_so(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model, _ = workflow_model([as_reply(MUL_TEST), server_error(), server_error(), server_error()])
    out = io.StringIO()
    code = await run_build(
        Plan(goal="g", behaviors=["mul"]), fixture_repo, model=model, model_id=MODEL, sandbox=LocalSandbox(),
        config=cli_config(tmp_path, fixture_venv), hints=False, ask=lambda _p: "yes", out=out, err=io.StringIO(),
    )
    assert code == API_ERROR_EXIT
    assert "status: api_error" in out.getvalue() and "InternalServerError 500 (request req_4)" in out.getvalue()
    [summary] = (fixture_repo / "runs").glob("*.json")
    assert json.loads(summary.read_text())["status"] == "api_error"


def test_the_client_gets_max_retries_and_timeout_from_config() -> None:
    client = make_client(AnthropicConfig(max_retries=5, timeout=30.0))
    assert client.max_retries == 5 and isinstance(client.timeout, httpx2.Timeout) and client.timeout.read == 30.0
    model = make_model(HarnessConfig(), choose(HarnessConfig(), model=MODEL), None)
    assert isinstance(model, AnthropicWorkflowModel) and model.client.max_retries == 2  # the SDK's default


# --- bench: an api_error run is counted and left out of the Green means --------------------------------------------------

def ledger(label: str, repeat: int, kata: str, green: int, status: str = "kept") -> dict[str, Any]:
    return {"label": label, "repeat": repeat, "kata": kata, "pair": kata, "summary": {
        "status": status, "planned": 6, "stuck": 0, "totals": {"green": green, "attempts": 6, "satisfied": 0, "seconds": 60, "cost_usd": 0.01},
    }}


def test_an_api_error_run_is_counted_in_the_report_and_left_out_of_the_green_means() -> None:
    lines = [ledger("a", 1, "bowling", 6), ledger("a", 2, "bowling", 0, "api_error"), ledger("a", 3, "bowling", 4)]
    stats = bench.aggregate(lines, "a")
    assert stats["bowling"].n == 2 and stats["bowling"].green == 5.0 and stats["bowling"].green_min == 4
    assert "1 run(s) ended with api_error" in bench.label_table(lines, "a")
    assert "a: 1 run(s) ended with api_error" in bench.compare_table([*lines, ledger("b", 1, "bowling", 6)], "a", "b")
    assert "api_error" not in bench.label_table(lines[:1], "a")


# --- gate 3: the chat turn ends with a note, and the history is cut back -------------------------------------------------

@pytest.mark.parametrize("script", [
    [streamed(text("Hello there, this is long"), cut_after=4)],  # ends cleanly, before message_stop
    [streamed(text("Hello there, this is long"), cut_after=4, reset=True)],  # a connection reset mid-stream
    [overloaded()] * 3,  # 529 on every attempt, past max_retries=2
], ids=["cut", "reset", "529"])
async def test_a_failed_chat_turn_leaves_the_session_consistent_and_the_next_turn_works(tmp_path: Path, script: list[Step]) -> None:
    transport = FaultyTransport([*script, streamed(text("after"))])
    app = HarnessApp(app_config(tmp_path), anthropic_client=transport.client())
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "hi")
        first = [r.text for r in app.query(Reply)][-1]
        assert "[API error:" in first and "the turn was dropped" in first
        assert app._messages == [] and app.meter is not None and not app.meter.state.in_flight and app.meter.state.calls == 0
        await send(app, pilot, "again")
        assert [r.text for r in app.query(Reply)][-1] == "after"
        # no half reply and no unanswered prompt went with the next turn
        assert transport.bodies[-1]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "again"}]}]
        assert app.store.messages(app.session_id)[-1]["content"][0]["text"] == "after"


# --- gate 4: max_tokens and refusal keep their F1 behavior ---------------------------------------------------------------

async def test_max_tokens_gets_one_retry_and_twice_raises_with_the_model_error() -> None:
    model, transport = workflow_model([reply('{"value": 3', "max_tokens"), reply(GOOD)])
    assert await model.complete("sys", "go", Answer) == Answer(value=3, why="because")
    assert len(transport.bodies) == 2 and "briefly" in transport.bodies[1]["messages"][0]["content"]
    model, _ = workflow_model([reply(GOOD, "max_tokens"), reply(GOOD, "max_tokens")])
    with pytest.raises(WorkflowModelError):
        await model.complete("sys", "go", Answer)


async def test_a_refusal_raises_with_its_category_and_is_not_an_api_failure() -> None:
    model, transport = workflow_model([reply("", "refusal", {"type": "refusal", "category": "cyber", "explanation": None})])
    with pytest.raises(WorkflowModelError, match="cyber") as caught:
        await model.complete("sys", "go", Answer)
    assert not isinstance(caught.value, ApiFailure) and len(transport.bodies) == 1


# --- gate 5: no API key in any error log line -----------------------------------------------------------------------------

async def test_no_api_key_appears_in_any_error_line(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    transport = FaultyTransport([as_reply(MUL_TEST), unauthorized()])  # the 401's message echoes the key it was sent
    model = AnthropicWorkflowModel(transport.client(api_key=None), MODEL)
    out, err = io.StringIO(), io.StringIO()
    code = await run_build(
        Plan(goal="g", behaviors=["mul"]), fixture_repo, model=model, model_id=MODEL, sandbox=LocalSandbox(),
        config=cli_config(tmp_path, fixture_venv), hints=False, ask=lambda _p: "yes", out=out, err=err,
    )
    assert code == API_ERROR_EXIT and len(transport.bodies) == 2  # a 401 is not retried
    [log] = (tmp_path / "runs").glob("*.jsonl")
    line = read_log(log)[-1]
    assert line["node"] == "api_error" and line["error"] == "AuthenticationError" and "sk-ant-…" in line["message"]
    [summary] = (fixture_repo / "runs").glob("*.json")
    for blob in (log.read_text(), summary.read_text(), out.getvalue(), err.getvalue()):
        assert KEY not in blob


async def test_no_api_key_appears_in_the_chat_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    transport = FaultyTransport([unauthorized()])
    app = HarnessApp(app_config(tmp_path), anthropic_client=transport.client(api_key=None))
    async with app.run_test() as pilot:
        await until(pilot, lambda: app._agent is not None)
        await send(app, pilot, "hi")
        note = [r.text for r in app.query(Reply)][-1]
        assert "AuthenticationError 401" in note and KEY not in note
