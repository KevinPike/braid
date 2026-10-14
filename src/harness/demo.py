"""Offline demos: a scripted Anthropic server over ``httpx2.MockTransport``, so ``harness --demo <name>`` needs no key,
no network and no spend.

``--demo choice`` runs the chat agent against a model that answers every prompt by asking a Choice (ADR 0010) and then
says what you picked. Nothing leaves the machine and ``~/.braid`` is not touched: the stores live in a temp folder.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import httpx2

from harness.config import HarnessConfig, load_config

if TYPE_CHECKING:
    from harness.app import HarnessApp

DEMO_MODEL = "claude-haiku-5-5"
DEMOS = ("choice",)
_USAGE = {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

# What the scripted model asks: two questions, the second with previews, as in the spec's mockup.
CHOICE_INPUT: dict[str, Any] = {"questions": [
    {"header": "Hashing", "question": "Which hash should the kata use for stored credentials?", "multiSelect": False, "options": [
        {"label": "argon2-cffi", "description": "memory-hard, the current default"},
        {"label": "bcrypt", "description": "older, widely deployed"},
        {"label": "scrypt", "description": "in the standard library"}]},
    {"header": "Storage", "question": "Where should the kata keep its users?", "multiSelect": False, "options": [
        {"label": "SQLite", "description": "one file, stdlib sqlite3",
         "preview": "CREATE TABLE users (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL\n);"},
        {"label": "JSON file", "description": "simplest, no queries", "preview": '{"users": [{"id": 1, "name": "ada"}]}'},
        {"label": "In memory", "description": "lost on exit; tests only"}]},
]}


def _event(kind: str, data: dict[str, Any]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()


def _stream(block: dict[str, Any], stop_reason: str) -> bytes:
    """One streamed message holding one text or tool_use block."""
    out = [_event("message_start", {"message": {
        "id": "msg_demo", "type": "message", "role": "assistant", "model": DEMO_MODEL, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {**_USAGE, "output_tokens": 1}}})]
    if block["type"] == "text":
        out.append(_event("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}))
        out.append(_event("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": block["text"]}}))
    else:
        out.append(_event("content_block_start", {"index": 0, "content_block": {**block, "input": {}}}))
        out.append(_event("content_block_delta", {"index": 0, "delta": {
            "type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
    out.append(_event("content_block_stop", {"index": 0}))
    out.append(_event("message_delta", {"delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": None},
                                        "usage": {"output_tokens": _USAGE["output_tokens"]}}))
    out.append(_event("message_stop", {}))
    return b"".join(out)


def _tool_result_text(body: dict[str, Any]) -> str | None:
    """The text of the tool result the last user message carries, or None when it is an ordinary prompt."""
    content = body["messages"][-1]["content"]
    for block in content if isinstance(content, list) else ():
        if block.get("type") == "tool_result":
            return "".join(c.get("text", "") for c in block.get("content") or ())
    return None


def choice_handler(request: httpx2.Request) -> httpx2.Response:
    """Every prompt gets a Choice; the tool result gets a sentence that names what was picked."""
    body = json.loads(request.content)
    answered = _tool_result_text(body)
    if answered is None:
        block: dict[str, Any] = {"type": "tool_use", "id": f"toolu_demo_{len(body['messages'])}",
                                 "name": "ask_user_question", "input": CHOICE_INPUT}
        stop = "tool_use"
    else:
        try:
            picked = json.loads(answered)
        except ValueError:
            picked = {}
        mapping = picked.get("answers")
        said = "; ".join(f"{q} {a}" for q, a in mapping.items()) if mapping else f"no answer ({picked.get('status', 'unknown')})"
        block, stop = {"type": "text", "text": f"(scripted demo) You chose: {said}. Send any prompt to be asked again."}, "end_turn"
    if not body.get("stream"):  # a non-streamed call (the summarizer)
        return httpx2.Response(200, json={
            "id": "msg_demo", "type": "message", "role": "assistant", "model": DEMO_MODEL, "stop_reason": "end_turn",
            "stop_sequence": None, "content": [{"type": "text", "text": "(demo summary)"}], "usage": _USAGE})
    return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_demo"},
                           content=_stream(block, stop))


def scripted_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="demo-not-a-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0)


def demo_app(name: str, config_path: Path | None = None, data_dir: Path | None = None) -> HarnessApp:
    """The app wired to the scripted model ``name``; raises ``SystemExit`` for an unknown demo."""
    from harness.app import HarnessApp

    if name not in DEMOS:
        raise SystemExit(f"harness: unknown demo '{name}' (known: {', '.join(DEMOS)})")
    os.environ.setdefault("ANTHROPIC_API_KEY", "demo-not-a-key")  # the SDK is never asked to send it
    base = load_config(config_path)
    data = data_dir or Path(tempfile.mkdtemp(prefix="braid-demo-"))
    config: HarnessConfig = base.model_copy(update={
        "paths": base.paths.model_copy(update={"data_dir": data}),
        "profile": base.profile.model_copy(update={"backend": "anthropic", "model": DEMO_MODEL}),
        "tools": base.tools.model_copy(update={"shell": False, "write": False, "ask": True}),
        "ask": base.ask.model_copy(update={"timeout_s": 0.0}),
    })
    app = HarnessApp(config, anthropic_client=scripted_client(choice_handler))
    app._startup_notes.append("demo: a scripted model, offline. Send any prompt and it will ask you a Choice.")
    return app
