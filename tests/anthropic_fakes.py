"""A fake Anthropic Messages API for the chat agent: streamed (SSE) replies through ``httpx2.MockTransport``, no network."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2

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
