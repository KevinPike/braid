import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import BaseModel

from harness.guard.ollama_client import OllamaClient
from harness.workflow.model import GuardedWorkflowModel, WorkflowModelError
from harness.workflow.plan import Plan

HOST = "http://ollama.test"


class Answer(BaseModel):
    value: int
    why: str


def reply(content: str) -> httpx.Response:
    lines = [
        {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
        {"message": {"role": "assistant", "content": content}, "done": False},
        {"done": True, "message": {"role": "assistant", "content": ""}, "done_reason": "stop"},
    ]
    return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())


def make() -> GuardedWorkflowModel:
    client = OllamaClient(HOST, keep_alive="30m")
    client.pin("m", 4096)
    return GuardedWorkflowModel(client, "m")


@respx.mock
async def test_a_valid_reply_parses() -> None:
    respx.post(f"{HOST}/api/chat").mock(return_value=reply('{"value": 3, "why": "because"}'))
    assert await make().complete("sys", "go", Answer) == Answer(value=3, why="because")


@respx.mock
async def test_one_invalid_reply_then_a_valid_one_parses_after_exactly_two_requests() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(side_effect=[reply("not json"), reply('{"value": 1, "why": "x"}')])
    assert (await make().complete("sys", "go", Answer)).value == 1
    assert route.call_count == 2
    retry = json.loads(route.calls.last.request.content)["messages"][1]["content"]
    assert retry.startswith("go") and "invalid" in retry  # the error is appended to the prompt


@respx.mock
async def test_two_invalid_replies_raise() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(side_effect=[reply("{}"), reply('{"value": "x"}')])
    with pytest.raises(WorkflowModelError):
        await make().complete("sys", "go", Answer)
    assert route.call_count == 2


@respx.mock
async def test_every_request_carries_format_and_the_pinned_num_ctx() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(side_effect=[reply("bad"), reply('{"value": 1, "why": "x"}')])
    await make().complete("sys", "go", Answer)
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["format"] == Answer.model_json_schema()
        assert body["options"]["num_ctx"] == 4096


def test_plan_round_trips_and_needs_a_behavior(tmp_path: Path) -> None:
    plan = Plan(goal="g", behaviors=["b"], assumptions=[])
    plan.save(tmp_path / "p.json")
    assert Plan.load(tmp_path / "p.json") == plan
    with pytest.raises(ValueError):
        Plan(goal="g", behaviors=[])


@respx.mock
async def test_think_and_num_predict_are_sent_while_num_ctx_stays_pinned() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(return_value=reply('{"value": 1, "why": "x"}'))
    client = OllamaClient(HOST)
    client.pin("m", 4096)
    await GuardedWorkflowModel(client, "m", think=False, num_predict=512).complete("s", "go", Answer)
    body = json.loads(route.calls.last.request.content)
    assert body["think"] is False and body["options"]["num_predict"] == 512 and body["options"]["num_ctx"] == 4096


@respx.mock
async def test_think_and_num_predict_are_omitted_by_default() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(return_value=reply('{"value": 1, "why": "x"}'))
    await make().complete("s", "go", Answer)
    body = json.loads(route.calls.last.request.content)
    assert "think" not in body and "num_predict" not in body["options"]


def thinking_only() -> httpx.Response:
    lines = [{"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False}, {"done": True}]
    return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())


@respx.mock
async def test_a_thinking_only_reply_is_named_in_the_retry_and_the_error() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(side_effect=[thinking_only(), thinking_only()])
    with pytest.raises(WorkflowModelError, match="only thinking text"):
        await make().complete("sys", "go", Answer)
    assert "only thinking text" in json.loads(route.calls.last.request.content)["messages"][1]["content"]


def test_workflow_config_defaults_stop_thinking_and_cap_output() -> None:
    from harness.config import WorkflowConfig, load_config

    assert WorkflowConfig().think is False and WorkflowConfig().num_predict == 4096
    assert load_config(Path("braid.toml")).workflow.num_predict == 4096
