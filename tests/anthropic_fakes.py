"""A fake Anthropic Messages API for the chat agent: streamed (SSE) replies through ``httpx2.MockTransport``, no network.

``FaultyTransport`` (F8) serves a script of faults and healthy replies, to both the chat agent and Workflows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
from anthropic._models import FinalRequestOptions

MODEL = "claude-haiku-5-5"
USAGE = {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}


def text(s: str) -> dict[str, Any]:
    return {"type": "text", "text": s}


def thinking(signature: str = "sig-1") -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": signature}


def tool_use(name: str, args: dict[str, Any], id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": id, "name": name, "input": args}


def _event(kind: str, data: dict[str, Any]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()


def sse_events(
    blocks: list[dict[str, Any]], stop_reason: str = "end_turn", usage: dict[str, int] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> list[bytes]:
    """The SSE events of one streamed message whose content is ``blocks``; text arrives in two deltas."""
    u = usage or USAGE
    out = [_event("message_start", {"message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {**u, "output_tokens": 1},
    }})]
    for i, block in enumerate(blocks):
        kind = block["type"]
        if kind == "text":
            out.append(_event("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}}))
            half = len(block["text"]) // 2
            for part in (block["text"][:half], block["text"][half:]):
                out.append(_event("content_block_delta", {"index": i, "delta": {"type": "text_delta", "text": part}}))
        elif kind == "thinking":
            out.append(_event("content_block_start", {"index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
            out.append(_event("content_block_delta", {"index": i, "delta": {"type": "signature_delta", "signature": block["signature"]}}))
        elif kind == "tool_use":
            out.append(_event("content_block_start", {"index": i, "content_block": {**block, "input": {}}}))
            out.append(_event("content_block_delta", {"index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        out.append(_event("content_block_stop", {"index": i}))
    out.append(_event("message_delta", {
        "delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details},
        "usage": {"output_tokens": u["output_tokens"]},
    }))
    out.append(_event("message_stop", {}))
    return out


@dataclass
class FakeApi:
    """Serves ``replies`` in order (each a list of SSE event bytes) and keeps every request body.

    ``hang_after`` holds the reply open after that many events, until the request is cancelled; ``closed`` counts the
    response bodies that were closed, which is how a cancelled stream shows."""

    replies: list[list[bytes]] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    hang_after: int | None = None
    closed: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)
    on_request: Callable[[dict[str, Any]], None] | None = None

    def reply(self, *blocks: dict[str, Any], **kw: Any) -> FakeApi:
        self.replies.append(sse_events(list(blocks), **kw))
        return self

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if not body.get("stream"):  # a non-streamed call (the summarizer): answer with the whole message at once
            self.bodies.append(body)
            return httpx2.Response(200, json=_message(self.replies[len(self.bodies) - 1]), headers={"request-id": f"req_{len(self.bodies)}"})
        self.bodies.append(body)
        if self.on_request is not None:
            self.on_request(body)
        events = self.replies[len(self.bodies) - 1]
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream", "request-id": f"req_{len(self.bodies)}"},
            content=self._body(events),
        )

    async def _body(self, events: list[bytes]) -> AsyncIterator[bytes]:
        try:
            for i, event in enumerate(events):
                if self.hang_after is not None and i == self.hang_after:
                    self.started.set()
                    await asyncio.Event().wait()  # until cancelled
                yield event
        finally:
            self.closed += 1

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handler)), max_retries=0,
        )


class InstantRetries(anthropic.AsyncAnthropic):
    """The real client, retrying as the SDK does but without waiting: ``waits`` keeps how long each retry would have
    slept, so a test can see ``retry-after`` honored."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.waits: list[float] = []

    def _calculate_retry_timeout(
        self, remaining_retries: int, options: FinalRequestOptions, response_headers: httpx2.Headers | None = None,
    ) -> float:
        self.waits.append(super()._calculate_retry_timeout(remaining_retries, options, response_headers))
        return 0.0


# --- F8: fault injection --------------------------------------------------------------------------------------------

Step = Callable[[httpx2.Request, int], httpx2.Response]  # (the request, its 1-based number) -> the response


def _error(status: int, kind: str, message: str, n: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status, json={"type": "error", "error": {"type": kind, "message": message}},
        headers={"request-id": f"req_{n}", **(headers or {})},
    )


def rate_limited(retry_after: str = "7") -> Step:
    return lambda _r, n: _error(429, "rate_limit_error", "Number of request tokens has exceeded your rate limit", n, {"retry-after": retry_after})


def overloaded() -> Step:
    return lambda _r, n: _error(529, "overloaded_error", "Overloaded", n)


def server_error() -> Step:
    return lambda _r, n: _error(500, "api_error", "Internal server error", n)


def unauthorized() -> Step:
    """A 401 whose message echoes the key it was sent, as a misbehaving proxy might. The SDK does not retry it."""
    return lambda r, n: _error(401, "authentication_error", f"invalid x-api-key: {r.headers.get('x-api-key')}", n)


def dropped() -> Step:
    """The connection drops before any response: the SDK sees a transport error and retries it."""
    def step(request: httpx2.Request, _n: int) -> httpx2.Response:
        raise httpx2.RemoteProtocolError("peer closed connection without sending complete message body", request=request)

    return step


def not_json() -> Step:
    """A 200 that says it is JSON and is not (a proxy's error page)."""
    return lambda _r, n: httpx2.Response(
        200, content=b"<html>502 Bad Gateway</html>", headers={"content-type": "application/json", "request-id": f"req_{n}"},
    )


def reply(text_: str, stop_reason: str = "end_turn", stop_details: dict[str, Any] | None = None) -> Step:
    """A healthy non-streamed message (a Workflow call) whose content is ``text_``."""
    body = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": text_}] if text_ else [], "stop_reason": stop_reason, "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }
    return lambda _r, n: httpx2.Response(200, json=body, headers={"request-id": f"req_{n}"})


def streamed(*blocks: dict[str, Any], cut_after: int | None = None, reset: bool = False, **kw: Any) -> Step:
    """A streamed message (a chat call). ``cut_after`` ends the stream after that many events, before ``message_stop``:
    cleanly, or with a connection reset when ``reset``."""
    events = sse_events(list(blocks), **kw)

    async def body() -> AsyncIterator[bytes]:
        for event in events[:cut_after]:
            yield event
        if reset:
            raise httpx2.ReadError("connection reset by peer")

    return lambda _r, n: httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": f"req_{n}"}, content=body())


@dataclass
class FaultyTransport:
    """An ``httpx2.MockTransport`` that serves ``script`` in order, one step per HTTP request (an SDK retry is a request
    too), and keeps every request body."""

    script: list[Step]
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        n = len(self.bodies)
        assert n <= len(self.script), f"the fault script ran out at request {n}"
        return self.script[n - 1](request, n)

    def client(self, *, max_retries: int = 2, api_key: str | None = "test-key") -> InstantRetries:
        """``api_key=None`` leaves the SDK to read ``ANTHROPIC_API_KEY``, as the harness does."""
        return InstantRetries(
            api_key=api_key, max_retries=max_retries, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handler)),
        )


# --- F12: Message Batches --------------------------------------------------------------------------------------------

Answer = Callable[[str, dict[str, Any]], dict[str, Any]]  # (custom_id, params) -> the result, as ``succeeded`` etc. build it


def succeeded(text_: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    return {"type": "succeeded", "message": {
        "id": "msg_b", "type": "message", "role": "assistant", "model": MODEL, "content": [{"type": "text", "text": text_}],
        "stop_reason": "end_turn", "stop_sequence": None, "stop_details": None,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }}


def errored(kind: str = "invalid_request_error", message: str = "output_config.format: not supported here") -> dict[str, Any]:
    return {"type": "errored", "error": {"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_err"}}


def expired() -> dict[str, Any]:
    return {"type": "expired"}


@dataclass
class FakeBatches:
    """The Message Batches endpoints on an ``httpx2.MockTransport``: create, retrieve (``in_progress`` for ``polls``
    retrievals, then ``ended``), and the results as JSON Lines in reverse order, so a caller must match by ``custom_id``.
    ``answer`` gives each request's result; ``batches`` keeps every batch's requests."""

    answer: Answer
    polls: int = 1
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    retrievals: dict[int, int] = field(default_factory=dict)  # per batch
    cancelled: list[str] = field(default_factory=list)
    on_create: Callable[[list[dict[str, Any]]], None] | None = None

    def _batch(self, n: int, ended: bool) -> dict[str, Any]:
        count = len(self.batches[n])
        return {
            "id": f"msgbatch_{n}", "type": "message_batch", "processing_status": "ended" if ended else "in_progress",
            "created_at": "2026-10-11T00:00:00Z", "expires_at": "2026-10-12T00:00:00Z",
            "ended_at": "2026-10-11T00:05:00Z" if ended else None,
            "request_counts": {"processing": 0 if ended else count, "succeeded": count if ended else 0, "errored": 0,
                               "canceled": 0, "expired": 0},
            "results_url": f"https://api.anthropic.com/v1/messages/batches/msgbatch_{n}/results" if ended else None,
        }

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            requests = json.loads(request.content)["requests"]
            self.batches.append(requests)
            if self.on_create is not None:
                self.on_create(requests)
            return httpx2.Response(200, json=self._batch(len(self.batches) - 1, ended=self.polls == 0))
        n = int(path.split("msgbatch_")[1].split("/")[0])
        if path.endswith("/cancel"):
            self.cancelled.append(f"msgbatch_{n}")
            return httpx2.Response(200, json=self._batch(n, ended=False))
        if path.endswith("/results"):
            lines = [{"custom_id": r["custom_id"], "result": self.answer(r["custom_id"], r["params"])} for r in self.batches[n]]
            body = "".join(json.dumps(x) + "\n" for x in reversed(lines))
            return httpx2.Response(200, content=body.encode(), headers={"content-type": "application/binary"})
        self.retrievals[n] = self.retrievals.get(n, 0) + 1
        return httpx2.Response(200, json=self._batch(n, ended=self.retrievals[n] >= self.polls))

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handler)), max_retries=0,
        )


def _message(events: list[bytes]) -> dict[str, Any]:
    """The non-streamed message equivalent of an SSE reply: its text blocks and usage."""
    content: list[dict[str, Any]] = []
    usage: dict[str, Any] = dict(USAGE)
    for raw in events:
        data = json.loads(raw.decode().split("data: ", 1)[1])
        if data["type"] == "content_block_delta" and data["delta"]["type"] == "text_delta":
            if not content or content[-1]["index"] != data["index"]:
                content.append({"index": data["index"], "type": "text", "text": ""})
            content[-1]["text"] += data["delta"]["text"]
        elif data["type"] == "message_start":
            usage = {**data["message"]["usage"]}
        elif data["type"] == "message_delta":
            usage["output_tokens"] = data["usage"]["output_tokens"]
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "stop_reason": "end_turn",
        "stop_sequence": None, "content": [{"type": "text", "text": c["text"]} for c in content], "usage": usage,
    }
