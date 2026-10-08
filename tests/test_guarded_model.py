import json

import httpx
import respx
from strands import Agent, tool

from harness.agent import GuardedOllamaModel, stream_reply
from harness.guard.ollama_client import CallMetrics, OllamaClient

HOST = "http://ollama.test"


def ndjson(*objs: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, content="\n".join(json.dumps(o) for o in objs).encode())


def final(**extra: object) -> dict[str, object]:
    return {"done": True, "message": {"role": "assistant", "content": ""}, "done_reason": "stop",
            "prompt_eval_count": 33, "eval_count": 7, "eval_duration": 10**9, "load_duration": 0,
            "total_duration": 2 * 10**9, **extra}


def make(system_prompt: str = "sys", tools: list[object] | None = None) -> tuple[Agent, OllamaClient, list[tuple[str, CallMetrics]]]:
    client = OllamaClient(HOST, keep_alive="30m")
    client.pin("m", 4096)
    seen: list[tuple[str, CallMetrics]] = []
    client.on_call(lambda model, metrics, est: seen.append((model, metrics)))
    agent = Agent(model=GuardedOllamaModel(client, "m"), system_prompt=system_prompt, callback_handler=None,
                  tools=tools or [])
    return agent, client, seen


@respx.mock
async def test_agent_streams_through_guard_client_with_pinned_num_ctx() -> None:
    route = respx.post(f"{HOST}/api/chat").mock(return_value=ndjson(
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        final(),
    ))
    agent, _, seen = make()
    text = "".join([d async for d in stream_reply(agent, "hi")])
    assert text == "Hello"
    body = json.loads(route.calls.last.request.content)
    assert body["options"]["num_ctx"] == 4096 and body["keep_alive"] == "30m"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert seen[0][1].prompt_eval_count == 33  # the watchdog sees every call


@respx.mock
async def test_tool_calls_round_trip() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    calls = iter([
        ndjson({"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}]}, "done": False}, final()),
        ndjson({"message": {"role": "assistant", "content": "5"}, "done": False}, final()),
    ])
    route = respx.post(f"{HOST}/api/chat").mock(side_effect=lambda req: next(calls))
    agent, _, _ = make(tools=[add])
    text = "".join([d async for d in stream_reply(agent, "2+3?")])
    assert text.strip().endswith("5")
    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["tools"][0]["function"]["name"] == "add"
