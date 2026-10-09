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
