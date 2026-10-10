import ast
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from harness.workflow.model import AnthropicWorkflowModel, WorkflowModelError

MODEL = "claude-haiku-5-5"
GOOD = '{"value": 3, "why": "because"}'


class Answer(BaseModel):
    value: int
    why: str


def message(
    text: str, stop_reason: str = "end_turn", stop_details: dict[str, Any] | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": usage or {
            "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        },
    }


def make(replies: list[dict[str, Any]], **kw: Any) -> tuple[AnthropicWorkflowModel, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=replies[len(bodies) - 1], headers={"request-id": f"req_{len(bodies)}"})

    client = anthropic.AsyncAnthropic(
        api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0,
    )
    return AnthropicWorkflowModel(client, MODEL, **kw), bodies


async def test_a_valid_reply_parses() -> None:
    model, bodies = make([message(GOOD)])
    assert await model.complete("sys", "go", Answer) == Answer(value=3, why="because")
    assert len(bodies) == 1


async def test_an_invalid_reply_gets_one_retry_with_the_error() -> None:
    model, bodies = make([message('{"value": "x"}'), message(GOOD)])
    assert await model.complete("sys", "go", Answer) == Answer(value=3, why="because")
    assert len(bodies) == 2
    assert "invalid" in bodies[1]["messages"][0]["content"]


async def test_two_invalid_replies_raise() -> None:
    model, bodies = make([message("nope"), message("still nope")])
    with pytest.raises(WorkflowModelError):
        await model.complete("sys", "go", Answer)
    assert len(bodies) == 2


async def test_no_sampling_parameters_and_the_system_prompt_is_cached() -> None:
    model, bodies = make([message(GOOD)], effort="low")
    await model.complete("sys", "go", Answer)
    body = bodies[0]
    assert not {"temperature", "top_p", "top_k", "thinking"} & body.keys()
    assert body["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["messages"][-1]["role"] == "user"  # no assistant prefill


async def test_effort_is_omitted_unless_set() -> None:
    model, bodies = make([message(GOOD)])
    await model.complete("sys", "go", Answer)
    assert "effort" not in bodies[0]["output_config"]


async def test_a_refusal_raises_with_its_category_and_does_not_retry() -> None:
    reply = message("", "refusal", {"type": "refusal", "category": "cyber", "explanation": None})
    model, bodies = make([reply, message(GOOD)])
    with pytest.raises(WorkflowModelError, match="cyber"):
        await model.complete("sys", "go", Answer)
    assert len(bodies) == 1


async def test_a_refusal_without_details_still_raises() -> None:
    model, _ = make([message("", "refusal")])
    with pytest.raises(WorkflowModelError, match="refused"):
        await model.complete("sys", "go", Answer)


async def test_max_tokens_triggers_the_retry() -> None:
    model, bodies = make([message('{"value": 3, "wh', "max_tokens"), message(GOOD)])
    assert await model.complete("sys", "go", Answer) == Answer(value=3, why="because")
    assert len(bodies) == 2
    assert "briefly" in bodies[1]["messages"][0]["content"]


async def test_max_tokens_twice_raises() -> None:
    model, _ = make([message(GOOD, "max_tokens"), message(GOOD, "max_tokens")])
    with pytest.raises(WorkflowModelError):
        await model.complete("sys", "go", Answer)


async def test_usage_cost_and_request_id_are_recorded_per_call() -> None:
    usage = {
        "input_tokens": 1_000_000, "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000,
    }
    model, _ = make([message("bad", usage=usage), message(GOOD)])
    await model.complete("sys", "go", Answer)
    first, second = model.calls
    assert (first.input_tokens, first.output_tokens) == (1_000_000, 1_000_000)
    assert (first.cache_read_input_tokens, first.cache_creation_input_tokens) == (1_000_000, 1_000_000)
    assert first.cost_usd == pytest.approx(0.10 + 0.50 + 0.01 + 0.125)
    assert (first.request_id, second.request_id) == ("req_1", "req_2")
    assert second.cost_usd == pytest.approx((10 * 0.10 + 5 * 0.50) / 1_000_000)
    assert model.cost_usd == pytest.approx(first.cost_usd + second.cost_usd)


def test_an_unknown_model_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="no pricing"):
        AnthropicWorkflowModel(anthropic.AsyncAnthropic(api_key="k"), "claude-mystery")


def test_guard_imports_neither_anthropic_nor_strands() -> None:
    """Static import graph: ``harness.guard`` imports only itself and third-party code other than these two (ADR 0001, 0006)."""
    guard = Path(__file__).parents[2] / "src" / "harness" / "guard"
    files = list(guard.rglob("*.py"))
    assert files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            for name in names:
                assert name.split(".")[0] not in {"anthropic", "strands"}, f"{path.name} imports {name}"
                assert not name.startswith("harness.") or name.startswith("harness.guard"), f"{path.name}: {name}"
