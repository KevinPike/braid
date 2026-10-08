"""Calibration: measure KV bytes per token with two loads (ADR 0002).

Cached per Ollama version and model digest, because either changing can change the number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.guard.backend import Backend


class CalibrationUnreliable(RuntimeError):
    """``/api/ps`` size cannot be trusted, so measuring KV cost from it would cache garbage."""


@dataclass(frozen=True)
class Calibration:
    model: str
    digest: str
    ollama_version: str
    weights: int  # bytes, extrapolated to num_ctx = 0
    kv_per_token: float  # bytes


@dataclass(frozen=True)
class GapReport:
    ratio: float  # measured / upper bound
    within_25pct: bool

    @property
    def explain_needed(self) -> bool:
        """Gate 4: outside 25% of the formula needs an explanation (e.g. sliding-window layers)."""
        return not self.within_25pct


def gap_report(*, measured: float, upper_bound: float) -> GapReport:
    ratio = measured / upper_bound
    return GapReport(ratio=ratio, within_25pct=abs(1 - ratio) <= 0.25)


class Calibrator:
    def __init__(self, backend: Backend, cache_path: Path) -> None:
        self._backend = backend
        self._path = cache_path
        self._expected = 0

    async def calibrate(self, model: str, *, ctx_lo: int = 4096, ctx_hi: int = 16384) -> Calibration:
        version = await self._backend.version()
        tag = next((t for t in await self._backend.tags() if t.name == model), None)
        digest = tag.digest if tag else ""
        self._expected = tag.size if tag else 0
        key = f"{version}:{digest}"
        cache = self._read()
        if key in cache:
            return Calibration(**cache[key])

        size_lo = await self._load_size(model, ctx_lo)
        size_hi = await self._load_size(model, ctx_hi)
        await self._backend.unload(model)
        k = (size_hi - size_lo) / (ctx_hi - ctx_lo)
        result = Calibration(model, digest, version, weights=round(size_lo - ctx_lo * k), kv_per_token=k)
        cache[key] = asdict(result)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(cache, indent=2))
        return result

    async def _load_size(self, model: str, num_ctx: int) -> int:
        self._backend.pin(model, num_ctx)
        await self._backend.unload(model)  # a fresh load, so num_ctx actually takes effect
        await self._backend.warm(model)
        loaded = next((m for m in await self._backend.ps() if m.name == model), None)
        if loaded is None:
            raise RuntimeError(f"{model} not loaded after warm at num_ctx={num_ctx}")
        if loaded.spilled:
            await self._backend.unload(model)
            raise RuntimeError(f"{model} would spill at num_ctx={num_ctx}; cannot calibrate")
        if loaded.undercounts(expected_size=self._expected):
            await self._backend.unload(model)
            raise CalibrationUnreliable(
                f"/api/ps size {loaded.size} is far below {model}'s {self._expected} bytes "
                "(speculative draft model, ollama/ollama#17251); use the architecture-derived KV cost instead"
            )
        return loaded.size

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        cache: dict[str, dict[str, Any]] = json.loads(self._path.read_text())
        return cache
