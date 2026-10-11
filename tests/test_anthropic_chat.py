"""F1b: the chat agent on the Anthropic API (``AnthropicChatModel``), with the Messages API faked."""

import asyncio

import pytest
from strands import Agent, tool

from harness.agent import AnthropicChatModel, build_api_agent, stream_reply
from harness.context.ledger import Ledger
from harness.usage import UsageMeter
from tests.anthropic_fakes import MODEL, FakeApi, text, thinking, tool_use


def make(api: FakeApi, tools: list[object] | None = None, **kw: object) -> tuple[Agent, UsageMeter, Ledger]:
    meter = UsageMeter(MODEL, 1_000_000)
    ledger = Ledger(lambda _m: 1_000_000)
    model = AnthropicChatModel(api.client(), MODEL, meter=meter, ledger=ledger, **kw)  # type: ignore[arg-type]
    return build_api_agent(model, "sys", tools=tools or []), meter, ledger


async def test_a_streamed_text_reply_arrives_as_deltas() -> None:
    api = FakeApi().reply(thinking(), text("Hello there"))
    agent, meter, ledger = make(api)
    deltas = [d async for d in stream_reply(agent, "hi")]
    assert "".join(deltas) == "Hello there" and len(deltas) == 2
    assert agent.messages[-1]["role"] == "assistant"
    # usage reaches the meter and the Ledger: prompt = input + cache reads + cache writes
    assert meter.state.calls == 1 and meter.state.input_tokens == 1120 and meter.state.output_tokens == 30
    assert meter.state.cost_usd == pytest.approx((120 * 0.10 + 1000 * 0.01 + 30 * 0.50) / 1e6)
    assert ledger.records[-1].prompt_eval_count == 1120 and ledger.records[-1].model == MODEL


async def test_a_tool_use_round_trip_replays_the_call_its_result_and_the_thinking_block() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    api = FakeApi().reply(thinking("sig-a"), tool_use("add", {"a": 2, "b": 3}), stop_reason="tool_use").reply(text("5"))
    agent, meter, _ = make(api, tools=[add])
    reply = "".join([d async for d in stream_reply(agent, "2+3?")])
    assert reply.strip().endswith("5")
    assert len(api.bodies) == 2 and meter.state.calls == 2
    assert api.bodies[0]["tools"][0]["name"] == "add"
    assistant, result = api.bodies[1]["messages"][1:3]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-a"}  # passed back unchanged
    assert assistant["content"][1]["type"] == "tool_use" and assistant["content"][1]["input"] == {"a": 2, "b": 3}
    assert result["content"][0]["type"] == "tool_result" and result["content"][0]["content"][0]["text"] == "5"


async def test_no_sampling_parameters_and_the_system_prompt_and_history_are_cached() -> None:
    api = FakeApi().reply(text("a")).reply(text("b"))
    agent, _, _ = make(api, effort="low")
    for prompt in ("one", "two"):
        [d async for d in stream_reply(agent, prompt)]
    for body in api.bodies:
        assert not {"temperature", "top_p", "top_k", "thinking"} & body.keys()
        assert body["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
        assert body["cache_control"] == {"type": "ephemeral"}  # automatic caching of the history prefix
        assert body["output_config"] == {"effort": "low"}
        assert body["messages"][-1]["role"] == "user"  # no assistant prefill
    assert api.bodies[1]["messages"][:2] == [  # the second call starts with the first call's history, byte for byte
        {"role": "user", "content": [{"type": "text", "text": "one"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
    ]


async def test_a_refusal_ends_the_turn_with_a_note() -> None:
    api = FakeApi().reply(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None})
    agent, _, _ = make(api)
    reply = "".join([d async for d in stream_reply(agent, "x")])
    assert "declined" in reply and "cyber" in reply


async def test_cancelling_mid_stream_closes_the_request_and_leaves_no_half_message() -> None:
    api = FakeApi(hang_after=4).reply(text("Hello there, this is long"))
    agent, meter, ledger = make(api)
    seen: list[str] = []

    async def consume() -> None:
        async for d in stream_reply(agent, "hi"):
            seen.append(d)

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(api.started.wait(), 5)
    await asyncio.sleep(0.05)
    assert seen  # the first delta arrived before the stream hung
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.closed == 1  # the HTTP stream was closed, which cancels the request
    assert not meter.state.in_flight and meter.state.calls == 0
    assert not any(m["role"] == "assistant" for m in agent.messages)  # no partial reply in the history
    assert ledger.pending is not None  # the estimate stays unmeasured; nothing was recorded as a finished call
