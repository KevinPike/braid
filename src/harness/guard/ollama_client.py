"""The single client every Ollama request goes through.

It pins ``num_ctx`` and ``keep_alive`` per model so they cannot drift and trigger a Reload.
A second request path would defeat the Guard (see CONTEXT.md, Pinned num_ctx).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

NS = 1_000_000_000
PS_TRUST_RATIO = 0.5


class UnpinnedModelError(RuntimeError):
    """A request was made for a model with no pinned num_ctx."""


@dataclass(frozen=True)
class CallMetrics:
    prompt_eval_count: int
    eval_count: int
    eval_duration_s: float
    load_duration_s: float
    total_duration_s: float = field(default=0.0, compare=False)

    @property
    def tokens_per_second(self) -> float | None:
        return self.eval_count / self.eval_duration_s if self.eval_duration_s > 0 else None


@dataclass(frozen=True)
class ChatChunk:
    text: str = ""
    thinking: str = ""
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    metrics: CallMetrics | None = None


@dataclass(frozen=True)
class RunningModel:
    name: str
    digest: str
    size: int
    size_vram: int
    expires_at: datetime
    context_length: int

    @property
    def spilled(self) -> bool:
        return self.size_vram < self.size

    def undercounts(self, *, expected_size: int) -> bool:
        """True when ``size`` is under half of what the model should take.

        Ollama with a speculative draft model reports the draft's footprint in place of the
        target's (ollama/ollama#17251, fix pending in #17857), which also hides spill.
        """
        return expected_size > 0 and self.size < expected_size * PS_TRUST_RATIO

    @property
    def gpu_fraction(self) -> float:
        return self.size_vram / self.size if self.size else 1.0


@dataclass(frozen=True)
class ModelInfo:
    """Architecture fields from ``/api/show`` that determine KV cache cost."""

    layers: int
    kv_heads: tuple[int, ...]  # per layer; Gemma 4 12B mixes 8 (sliding) and 1 (global)
    key_length: int
    value_length: int
    trained_ctx: int
    sliding_pattern: tuple[bool, ...] = ()  # True = sliding-window layer; empty = all global
    shared_kv_layers: int = 0  # trailing layers that reuse earlier layers' KV and store none
    sliding_window: int = 0
    key_length_swa: int = 0
    value_length_swa: int = 0
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelTag:
    name: str
    digest: str
    size: int


# model, metrics, the caller's prompt-token estimate (None when it has none)
CallListener = Callable[[str, CallMetrics, int | None], None]
Message = Mapping[str, Any]


class OllamaClient:
    def __init__(self, host: str, *, keep_alive: str = "30m", http: httpx.AsyncClient | None = None) -> None:
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
        self._pins: dict[str, int] = {}
        self._listeners: list[CallListener] = []

    def pin(self, model: str, num_ctx: int) -> None:
        self._pins[model] = num_ctx

    def pinned(self, model: str) -> int | None:
        return self._pins.get(model)

    def on_call(self, listener: CallListener) -> None:
        self._listeners.append(listener)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        estimate: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        num_ctx = self._pins.get(model)
        if num_ctx is None:
            raise UnpinnedModelError(f"{model} has no pinned num_ctx; call pin() first")
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {**(options or {}), "num_ctx": num_ctx},
        }
        if tools:
            body["tools"] = list(tools)
        async with self._http.stream("POST", f"{self.host}/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    raise RuntimeError(f"ollama: {obj['error']}")
                msg = obj.get("message") or {}
                metrics = _metrics(obj) if obj.get("done") else None
                yield ChatChunk(
                    text=msg.get("content", ""),
                    thinking=msg.get("thinking", ""),
                    tool_calls=tuple(msg.get("tool_calls") or ()),
                    metrics=metrics,
                )
                if metrics is not None:
                    for listener in self._listeners:
                        listener(model, metrics, estimate)

    async def unload(self, model: str) -> None:
        resp = await self._http.post(f"{self.host}/api/generate", json={"model": model, "keep_alive": 0})
        resp.raise_for_status()

    async def warm(self, model: str) -> None:
        """Load ``model`` at its pinned num_ctx without generating."""
        num_ctx = self._pins.get(model)
        if num_ctx is None:
            raise UnpinnedModelError(f"{model} has no pinned num_ctx; call pin() first")
        resp = await self._http.post(
            f"{self.host}/api/generate",
            json={"model": model, "keep_alive": self.keep_alive, "options": {"num_ctx": num_ctx}},
        )
        resp.raise_for_status()

    async def refresh_keep_alive(self, model: str) -> None:
        await self.warm(model)

    async def version(self) -> str:
        resp = await self._http.get(f"{self.host}/api/version")
        resp.raise_for_status()
        return str(resp.json()["version"])

    async def ps(self) -> list[RunningModel]:
        resp = await self._http.get(f"{self.host}/api/ps")
        resp.raise_for_status()
        return [
            RunningModel(
                name=m["name"],
                digest=m.get("digest", ""),
                size=m["size"],
                size_vram=m["size_vram"],
                expires_at=datetime.fromisoformat(m["expires_at"]),
                context_length=m.get("context_length", 0),
            )
            for m in resp.json().get("models", [])
        ]

    async def tags(self) -> list[ModelTag]:
        resp = await self._http.get(f"{self.host}/api/tags")
        resp.raise_for_status()
        return [ModelTag(m["name"], m.get("digest", ""), m.get("size", 0)) for m in resp.json().get("models", [])]

    async def show(self, model: str) -> ModelInfo:
        resp = await self._http.post(f"{self.host}/api/show", json={"model": model})
        resp.raise_for_status()
        data = resp.json()
        info: Mapping[str, Any] = data["model_info"]
        arch = info["general.architecture"]
        heads = info[f"{arch}.attention.head_count"]
        attn = f"{arch}.attention"
        key = info.get(f"{attn}.key_length", info[f"{arch}.embedding_length"] // heads)
        return ModelInfo(
            layers=info[f"{arch}.block_count"],
            kv_heads=_per_layer(info.get(f"{attn}.head_count_kv", heads), info[f"{arch}.block_count"]),
            key_length=key,
            value_length=info.get(f"{attn}.value_length", key),
            trained_ctx=info[f"{arch}.context_length"],
            sliding_pattern=tuple(info.get(f"{attn}.sliding_window_pattern", ())),
            shared_kv_layers=info.get(f"{attn}.shared_kv_layers", 0),
            sliding_window=info.get(f"{attn}.sliding_window", 0),
            key_length_swa=info.get(f"{attn}.key_length_swa", 0),
            value_length_swa=info.get(f"{attn}.value_length_swa", 0),
            capabilities=tuple(data.get("capabilities", ())),
        )


def _metrics(obj: Mapping[str, Any]) -> CallMetrics:
    return CallMetrics(
        prompt_eval_count=obj.get("prompt_eval_count", 0),
        eval_count=obj.get("eval_count", 0),
        eval_duration_s=obj.get("eval_duration", 0) / NS,
        load_duration_s=obj.get("load_duration", 0) / NS,
        total_duration_s=obj.get("total_duration", 0) / NS,
    )


def _per_layer(value: int | Sequence[int], layers: int) -> tuple[int, ...]:
    return (value,) * layers if isinstance(value, int) else tuple(value)
