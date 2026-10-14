"""D1: a Tape records exchanges and replays them with no network, matched by request, with no credential kept."""

import asyncio
import json
from pathlib import Path

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from harness.anthropic_api import ApiFailure
from harness.replay import Tape, TapeDrift, fingerprint, recording_transport, replay_transport
from harness.workflow.model import AnthropicWorkflowModel
from tests.anthropic_fakes import MODEL, FakeApi, text

KEY = "sk-ant-api03-SECRETSECRETSECRET"


class Answer(BaseModel):
    value: int


class Counted(httpx2.AsyncBaseTransport):
    """A transport that answers by ``inner`` and counts the requests that reach it."""

    def __init__(self, inner: httpx2.MockTransport) -> None:
        self.inner = inner
        self.seen = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.seen += 1
        return await self.inner.handle_async_request(request)


def client_on(transport: httpx2.AsyncBaseTransport) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=KEY, http_client=httpx2.AsyncClient(transport=transport), max_retries=0)


def echo(delay: float = 0.0) -> httpx2.MockTransport:
    """Answers each message with its own prompt as the JSON ``value``, and with headers a Tape must not keep."""
    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(delay)
        prompt = json.loads(request.content)["messages"][0]["content"]
        reply = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "stop_reason": "end_turn",
            "stop_sequence": None, "stop_details": None,
            "content": [{"type": "text", "text": json.dumps({"value": int(prompt)})}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        }
        return httpx2.Response(
            200, json=reply, headers={"request-id": f"req_{prompt}", "set-cookie": "session=abc", "x-api-key": request.headers["x-api-key"]},
        )

    return httpx2.MockTransport(handler)


async def test_a_recorded_workflow_call_replays_identically(tmp_path: Path) -> None:
    live = Counted(echo())
    recorded = AnthropicWorkflowModel(client_on(recording_transport(live, tmp_path)), MODEL)
    first = await recorded.complete("sys", "7", Answer)

    live.seen = 0
    replayed = AnthropicWorkflowModel(client_on(replay_transport(Tape.load(tmp_path), pace=0)), MODEL)
    assert await replayed.complete("sys", "7", Answer) == first == Answer(value=7)
    assert live.seen == 0
    assert replayed.calls[0].request_id == recorded.calls[0].request_id == "req_7"


async def test_a_streamed_reply_replays_byte_for_byte(tmp_path: Path) -> None:
    api = FakeApi().reply(text("Hello, braid"))
    request = {"model": MODEL, "max_tokens": 5, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    async with httpx2.AsyncClient(transport=recording_transport(httpx2.MockTransport(api._handler), tmp_path)) as c:
        live = await c.post("https://api.anthropic.com/v1/messages", json=request)
    async with httpx2.AsyncClient(transport=replay_transport(Tape.load(tmp_path), pace=0)) as c:
        again = await c.post("https://api.anthropic.com/v1/messages", json=request)
    assert again.content == live.content != b""
    assert again.headers["content-type"] == "text/event-stream"
    assert len(api.bodies) == 1


async def test_parallel_requests_get_their_own_reply_in_any_order(tmp_path: Path) -> None:
    recorder = AnthropicWorkflowModel(client_on(recording_transport(echo(), tmp_path)), MODEL)
    await asyncio.gather(*(recorder.complete("critic", p, Answer) for p in ("1", "2", "3")))

    replayer = AnthropicWorkflowModel(client_on(replay_transport(Tape.load(tmp_path), pace=0)), MODEL)
    got = await asyncio.gather(*(replayer.complete("critic", p, Answer) for p in ("3", "1", "2")))
    assert [a.value for a in got] == [3, 1, 2]


async def test_an_unknown_request_is_drift_and_never_reaches_the_network(tmp_path: Path) -> None:
    live = Counted(echo())
    await AnthropicWorkflowModel(client_on(recording_transport(live, tmp_path)), MODEL).complete("sys", "7", Answer)
    live.seen = 0

    tape = Tape.load(tmp_path)
    with pytest.raises(TapeDrift, match="fingerprint"):
        async with httpx2.AsyncClient(transport=replay_transport(tape)) as c:
            await c.post("https://api.anthropic.com/v1/messages", json={"model": MODEL, "messages": []})
    model = AnthropicWorkflowModel(client_on(replay_transport(tape, pace=0)), MODEL)
    with pytest.raises(ApiFailure) as info:  # the SDK wraps what a transport raises, and the model wraps that
        await model.complete("sys", "8", Answer)
    assert isinstance(info.value.__cause__, anthropic.APIConnectionError)
    assert isinstance(info.value.__cause__.__cause__, TapeDrift)
    assert live.seen == 0


async def test_a_repeat_beyond_the_recording_is_drift_too(tmp_path: Path) -> None:
    await AnthropicWorkflowModel(client_on(recording_transport(echo(), tmp_path)), MODEL).complete("sys", "7", Answer)
    model = AnthropicWorkflowModel(client_on(replay_transport(Tape.load(tmp_path), pace=0)), MODEL)
    await model.complete("sys", "7", Answer)
    with pytest.raises(ApiFailure):
        await model.complete("sys", "7", Answer)


async def test_loose_replay_serves_the_next_exchange_and_lists_the_drift(tmp_path: Path) -> None:
    await AnthropicWorkflowModel(client_on(recording_transport(echo(), tmp_path)), MODEL).complete("sys", "7", Answer)
    transport = replay_transport(Tape.load(tmp_path), pace=0, strict=False)
    reply = await AnthropicWorkflowModel(client_on(transport), MODEL).complete("sys", "8", Answer)
    assert reply == Answer(value=7)
    assert len(transport.drifts) == 1


def test_the_fingerprint_ignores_volatile_fields_and_key_order() -> None:
    a = json.dumps({"model": "m", "messages": [1], "session_id": "x", "metadata": {"u": 1}}).encode()
    b = json.dumps({"messages": [1], "model": "m", "session_id": "y"}).encode()
    c = json.dumps({"model": "m", "messages": [2]}).encode()
    assert fingerprint("POST", "/v1/messages", a) == fingerprint("POST", "/v1/messages", b)
    assert fingerprint("POST", "/v1/messages", a) != fingerprint("POST", "/v1/messages", c)
    assert fingerprint("GET", "/v1/key", b"") != fingerprint("GET", "/v1/models", b"")


async def test_a_catalog_style_get_replays(tmp_path: Path) -> None:
    handler = httpx2.MockTransport(lambda r: httpx2.Response(200, json={"data": [{"id": "a/b"}]}))
    async with httpx2.AsyncClient(transport=recording_transport(handler, tmp_path)) as c:
        live = await c.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    async with httpx2.AsyncClient(transport=replay_transport(Tape.load(tmp_path), pace=0)) as c:
        assert (await c.get("https://openrouter.ai/api/v1/models")).json() == live.json()


async def test_no_key_or_credential_header_survives_in_a_tape(tmp_path: Path) -> None:
    def leaky(request: httpx2.Request) -> httpx2.Response:  # a misbehaving proxy that echoes the key in the body
        return httpx2.Response(
            401, json={"error": f"invalid x-api-key: {request.headers['x-api-key']}, Bearer {KEY}"},
            headers={"request-id": "req_1", "set-cookie": "s=1", "authorization": f"Bearer {KEY}", "x-api-key": KEY},
        )

    client = client_on(recording_transport(httpx2.MockTransport(leaky), tmp_path))
    with pytest.raises(anthropic.AuthenticationError):
        await client.messages.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": f"my key is {KEY}"}])
    await AnthropicWorkflowModel(client_on(recording_transport(echo(), tmp_path)), MODEL).complete("sys", "7", Answer)

    saved = (tmp_path / "exchanges.jsonl").read_text()
    assert saved.count("\n") == 2
    for forbidden in ("sk-", "SECRET", "authorization", "cookie", "Bearer"):
        assert forbidden.lower() not in saved.lower()
    for row in saved.splitlines():  # no credential header is stored (the proxy's message may name one, redacted)
        assert set(json.loads(row)["headers"]) <= {"content-type", "request-id", "x-request-id", "retry-after"}
    assert "req_1" in saved  # request ids stay


async def test_pace_scales_the_waiting(tmp_path: Path) -> None:
    await AnthropicWorkflowModel(client_on(recording_transport(echo(0.4), tmp_path)), MODEL).complete("sys", "7", Answer)
    tape = Tape.load(tmp_path)
    assert tape.exchanges[0].seconds >= 0.4

    seconds = []
    for pace in (1.0, 0.5, 0.0):
        model = AnthropicWorkflowModel(client_on(replay_transport(tape, pace=pace)), MODEL)
        await model.complete("sys", "7", Answer)
        seconds.append(model.calls[0].seconds)
    full, half, none = seconds
    assert full is not None and half is not None and none is not None
    assert half == pytest.approx(full / 2, abs=0.1)
    assert none < 0.1


def test_a_tape_locates_its_sidecars(tmp_path: Path) -> None:
    (tmp_path / "meta.toml").write_text('commit = "abc"\n')
    tape = Tape.load(tmp_path)
    assert tape.exchanges == () and tape.meta == {"commit": "abc"}
    assert tape.catalog_path == tmp_path / "catalog.json" and tape.ledger_path == tmp_path / "ledger.jsonl"
