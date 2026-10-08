from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from harness.guard.ollama_client import ModelInfo, ModelTag, RunningModel

GIB = 1024**3


class FakeBackend:
    """In-memory Ollama: loaded size = weights + num_ctx * kv_per_token, spilling past ``gpu_bytes``."""

    def __init__(
        self,
        *,
        models: dict[str, tuple[int, float]],  # name -> (weights, kv_per_token)
        gpu_bytes: int = 100 * GIB,
        version: str = "0.35.1",
        trained_ctx: int = 131072,
        clock: Callable[[], datetime] | None = None,
        draft_only_ps: bool = False,
    ) -> None:
        # ollama#17251: with a speculative draft model, /api/ps reports only the draft's small footprint.
        self.draft_only_ps = draft_only_ps
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.keep_alive_s = 1800.0
        self.models = models
        self.gpu_bytes = gpu_bytes
        self._version = version
        self.trained_ctx = trained_ctx
        self.pins: dict[str, int] = {}
        self.loaded: dict[str, int] = {}  # name -> num_ctx at load
        self.load_log: list[tuple[str, int]] = []

    def pin(self, model: str, num_ctx: int) -> None:
        self.pins[model] = num_ctx

    async def version(self) -> str:
        return self._version

    async def tags(self) -> list[ModelTag]:
        return [ModelTag(n, f"digest-{n}", w) for n, (w, _) in self.models.items()]

    async def show(self, model: str) -> ModelInfo:
        _, k = self.models[model]
        # One global layer, one kv head: k = (key + value) * 2 bytes.
        return ModelInfo(layers=1, kv_heads=(1,), key_length=int(k // 4), value_length=int(k // 4), trained_ctx=self.trained_ctx)

    async def warm(self, model: str) -> None:
        self.loaded[model] = self.pins[model]
        self.load_log.append((model, self.pins[model]))

    async def unload(self, model: str) -> None:
        self.loaded.pop(model, None)

    async def ps(self) -> list[RunningModel]:
        out: list[RunningModel] = []
        used = 0
        for name, ctx in self.loaded.items():
            w, k = self.models[name]
            size = int(w + ctx * k)
            if self.draft_only_ps:
                size = int(size * 0.05)
            vram = size if self.draft_only_ps else min(size, max(0, self.gpu_bytes - used))
            used += vram
            out.append(RunningModel(
                name, f"digest-{name}", size, vram,
                self.clock() + timedelta(seconds=self.keep_alive_s), ctx,
            ))
        return out
