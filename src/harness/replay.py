"""The Tape: recorded API exchanges that replay serves instead of the network (specs/tui-demo.md, D1).

The seam is the httpx2 transport. Every model call (chat, Anthropic and OpenRouter Workflows) and the Catalog and key
calls go through an ``httpx2`` client, so a replay transport fakes them all with no change to the clients above it, and a
recording transport wraps the real one. A Tape is a directory: ``exchanges.jsonl`` (this module), plus ``catalog.json``,
``ledger.jsonl`` and ``meta.toml`` for the demo, which only ``Tape`` locates.

A request is matched by a fingerprint of its method, path and normalized body, never by arrival order, so parallel
Critiques each get their own reply. Replay never forwards: a request the Tape cannot answer raises ``TapeDrift``.
Recording keeps no credentials: only allowlisted response headers are stored, and key patterns are redacted.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
import tomllib
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx2

EXCHANGES = "exchanges.jsonl"
_VOLATILE = ("session_id", "metadata")  # request fields that differ run to run without changing the question asked
_KEPT_HEADERS = ("content-type", "request-id", "x-request-id", "retry-after")  # an allowlist: nothing else is stored
_SECRET = re.compile(r"\bsk-[A-Za-z0-9_\-]+|(?i:bearer)\s+[A-Za-z0-9_.\-]+")


class TapeDrift(Exception):
    """A request the Tape has no answer for. It is never forwarded to the network."""


def redact(text: str) -> str:
    """``text`` with API keys (``sk-ant-…``, ``sk-or-…``) and bearer tokens replaced."""
    return _SECRET.sub("[redacted]", text)


def fingerprint(method: str, path: str, body: bytes) -> str:
    """A short hash of what a request asks: method, path and query, and the body minus its volatile fields (key order and
    whitespace do not matter)."""
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        normal = json.dumps({k: v for k, v in parsed.items() if k not in _VOLATILE}, sort_keys=True, ensure_ascii=False)
    else:
        normal = body.decode("utf-8", "replace")
    digest = hashlib.sha256(f"{method} {redact(path)}\n{redact(normal)}".encode()).hexdigest()
    return digest[:16]


def _model_of(body: bytes) -> str:
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        return ""
    return str(parsed.get("model", "")) if isinstance(parsed, dict) else ""


def _target(request: httpx2.Request) -> str:
    return request.url.raw_path.decode()  # path and query, no host


@dataclass(frozen=True)
class Exchange:
    """One recorded request and its response. ``chunks`` are the body as streamed (utf-8 text, or base64 when
    ``encoding`` is ``"b64"``); ``seconds`` is the recorded wall time of the whole exchange."""

    fingerprint: str
    method: str
    path: str
    model: str
    status: int
    headers: dict[str, str]
    chunks: tuple[str, ...]
    seconds: float
    encoding: str = "utf8"

    def body_chunks(self) -> list[bytes]:
        return [base64.b64decode(c) if self.encoding == "b64" else c.encode() for c in self.chunks]


@dataclass(frozen=True)
class Tape:
    path: Path
    exchanges: tuple[Exchange, ...]

    @classmethod
    def load(cls, path: Path) -> Tape:
        """The Tape in directory ``path``; a missing ``exchanges.jsonl`` is an empty Tape."""
        file = path / EXCHANGES
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.exists() else []
        return cls(path, tuple(Exchange(**{**r, "chunks": tuple(r["chunks"])}) for r in rows))

    @property
    def catalog_path(self) -> Path:
        return self.path / "catalog.json"

    @property
    def ledger_path(self) -> Path:
        return self.path / "ledger.jsonl"

    @property
    def meta(self) -> dict[str, Any]:
        try:
            return tomllib.loads((self.path / "meta.toml").read_text())
        except FileNotFoundError:
            return {}


class _Chunks(httpx2.AsyncByteStream):
    """A body served in chunks, spread over ``delay`` seconds so a stream visibly types."""

    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self.chunks = chunks
        self.delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay > 0:
                await asyncio.sleep(self.delay / len(self.chunks))
            yield chunk


class ReplayTransport(httpx2.AsyncBaseTransport):
    """Serves a Tape. ``drifts`` lists, in loose mode, the requests answered from the wrong place."""

    def __init__(self, tape: Tape, *, pace: float, strict: bool) -> None:
        self.pace = pace
        self.strict = strict
        self.drifts: list[str] = []
        self._queues: dict[str, deque[Exchange]] = defaultdict(deque)
        for exchange in tape.exchanges:
            self._queues[exchange.fingerprint].append(exchange)
        self._order = list(tape.exchanges)
        self._taken: set[int] = set()

    def _take(self, exchange: Exchange) -> Exchange:
        self._taken.add(id(exchange))
        return exchange

    def _find(self, key: str, method: str, target: str, model: str) -> Exchange:
        queue = self._queues.get(key)
        while queue:
            exchange = queue.popleft()
            if id(exchange) not in self._taken:
                return self._take(exchange)
        what = f"{method} {redact(target)} (model {model or '-'}, fingerprint {key})"
        if self.strict:
            raise TapeDrift(f"no exchange on the Tape for {what}")
        for exchange in self._order:  # loose: the next unused exchange for this model and path
            if id(exchange) not in self._taken and exchange.model == model and exchange.path == target:
                self.drifts.append(what)
                return self._take(exchange)
        raise TapeDrift(f"no exchange left on the Tape for {what}")

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        target = _target(request)
        exchange = self._find(fingerprint(request.method, target, body), request.method, target, _model_of(body))
        return httpx2.Response(
            exchange.status, headers=exchange.headers, stream=_Chunks(exchange.body_chunks(), exchange.seconds * self.pace),
            request=request,
        )


def replay_transport(tape: Tape, *, pace: float = 0.5, strict: bool = True) -> ReplayTransport:
    """A transport that answers from ``tape`` and never reaches the network. It sleeps for the recorded seconds times
    ``pace`` (0: no waiting), spread across a streamed body's chunks. ``strict`` raises ``TapeDrift`` for a request with
    no exchange; loose mode serves the next unused exchange for that model and path, and lists the drift."""
    return ReplayTransport(tape, pace=pace, strict=strict)


class _Tee(httpx2.AsyncByteStream):
    """Passes the response body through and, once it has all been read, writes the exchange to the Tape."""

    def __init__(self, source: httpx2.Response, write: Any) -> None:
        self.source = source
        self.write = write
        self.chunks: list[bytes] = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.source.aiter_bytes():  # decoded: the stored headers say nothing of content-encoding
            self.chunks.append(chunk)
            yield chunk
        self.write(self.chunks)  # a stream cut short is never written

    async def aclose(self) -> None:
        await self.source.aclose()


@dataclass
class RecordingTransport(httpx2.AsyncBaseTransport):
    """Forwards to ``inner`` and appends each completed exchange to ``out/exchanges.jsonl``, stripped of credentials."""

    inner: httpx2.AsyncBaseTransport
    out: Path

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        target = _target(request)
        start = time.monotonic()
        response = await self.inner.handle_async_request(request)
        kept = {k: v for k, v in response.headers.items() if k.lower() in _KEPT_HEADERS}

        def write(chunks: list[bytes]) -> None:
            seconds = round(time.monotonic() - start, 3)
            try:
                texts = [redact(c.decode()) for c in chunks]
                encoding = "utf8"
            except UnicodeDecodeError:  # not text, so there is no key pattern to find
                texts = [base64.b64encode(c).decode() for c in chunks]
                encoding = "b64"
            exchange = Exchange(
                fingerprint(request.method, target, body), request.method, redact(target), _model_of(body),
                response.status_code, {k: redact(v) for k, v in kept.items()}, tuple(texts), seconds, encoding,
            )
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / EXCHANGES).open("a") as f:
                f.write(json.dumps(asdict(exchange), ensure_ascii=False) + "\n")

        return httpx2.Response(response.status_code, headers=kept, stream=_Tee(response, write), request=request)


def recording_transport(inner: httpx2.AsyncBaseTransport, out: Path) -> RecordingTransport:
    """``inner`` with every completed exchange written to the Tape directory ``out``. Request bodies are not stored, only
    their fingerprint, and no credential survives: headers are an allowlist and key patterns are redacted."""
    return RecordingTransport(inner, out)


__all__ = ["Exchange", "RecordingTransport", "ReplayTransport", "Tape", "TapeDrift", "fingerprint", "recording_transport", "redact", "replay_transport"]
