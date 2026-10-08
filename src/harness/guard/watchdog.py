"""Watchdog: the in-session half of the Guard.

A background ``/api/ps`` poller plus per-call metric checks; each finding is an Alert.
The checks are pure functions so the watchdog table can be tested row by row.
"""

from __future__ import annotations

import asyncio
import statistics
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Protocol

from harness.guard.ollama_client import CallMetrics, RunningModel

TRUNCATION_TOLERANCE = 0.05
RELOAD_THRESHOLD_S = 1.0
EVICTION_WINDOW_S = 60.0
SPEED_DROP_RATIO = 0.5
MIN_SPEED_SAMPLES = 3
CALL_ALERT_TTL = timedelta(seconds=60)


class Level(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2


@dataclass(frozen=True)
class Alert:
    kind: str
    level: Level
    message: str


@dataclass(frozen=True)
class GuardState:
    model: str | None = None
    gpu_fraction: float | None = None
    size: int | None = None
    num_ctx: int | None = None
    tokens_per_second: float | None = None
    keep_alive_s: float | None = None
    memory_level: int = 1  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
    alerts: tuple[Alert, ...] = ()

    @property
    def level(self) -> Level:
        return max((a.level for a in self.alerts), default=Level.GREEN)


def check_ps(models: Sequence[RunningModel], *, pinned: Mapping[str, int], now: datetime) -> list[Alert]:
    alerts: list[Alert] = []
    for m in models:
        if m.spilled:
            alerts.append(Alert("spill", Level.RED, f"{m.name} spilled to CPU: {m.gpu_fraction:.0%} on GPU"))
        want = pinned.get(m.name)
        if want is not None and m.context_length and m.context_length != want:
            alerts.append(Alert(
                "num_ctx_drift", Level.RED,
                f"{m.name} loaded at num_ctx {m.context_length}, pinned {want}: reloaded by a request outside the guard client",
            ))
        if (m.expires_at - now).total_seconds() < EVICTION_WINDOW_S:
            alerts.append(Alert("eviction_risk", Level.YELLOW, f"{m.name} unloads in under {EVICTION_WINDOW_S:.0f}s"))
    return alerts


def check_ps_trust(models: Sequence[RunningModel], *, expected: Mapping[str, int]) -> list[Alert]:
    return [
        Alert(
            "ps_unreliable", Level.YELLOW,
            f"/api/ps undercounts {m.name} ({m.size / 2**30:.1f} GiB vs ~{expected[m.name] / 2**30:.1f}); "
            "spill detection unreliable (ollama#17251)",
        )
        for m in models
        if m.name in expected and m.undercounts(expected_size=expected[m.name])
    ]


def check_memory(level: int) -> list[Alert]:
    if level >= 4:
        return [Alert("memory_pressure", Level.RED, "macOS memory pressure critical")]
    if level >= 2:
        return [Alert("memory_pressure", Level.YELLOW, "macOS memory pressure elevated")]
    return []


def read_memory_level() -> int:
    """macOS memory pressure via sysctl; 1 (normal) if unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, check=True
        )
        return int(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 1


def check_call(
    metrics: CallMetrics, *, estimate: int | None, first_call: bool, speeds: Sequence[float]
) -> list[Alert]:
    alerts: list[Alert] = []
    if estimate is not None and metrics.prompt_eval_count < estimate * (1 - TRUNCATION_TOLERANCE):
        gap = estimate - metrics.prompt_eval_count
        alerts.append(Alert(
            "truncation", Level.RED,
            f"prompt truncated: Ollama evaluated {metrics.prompt_eval_count} tokens, estimate {estimate} (gap {gap})",
        ))
    if not first_call and metrics.load_duration_s > RELOAD_THRESHOLD_S:
        alerts.append(Alert(
            "reload", Level.YELLOW,
            f"model reloaded ({metrics.load_duration_s:.1f}s load); check num_ctx drift or eviction",
        ))
    tps = metrics.tokens_per_second
    if tps is not None and len(speeds) >= MIN_SPEED_SAMPLES and tps < SPEED_DROP_RATIO * statistics.median(speeds):
        alerts.append(Alert(
            "speed_drop", Level.YELLOW,
            f"{tps:.0f} tok/s, under half the session median {statistics.median(speeds):.0f}; likely spill or pressure",
        ))
    return alerts


class WatchdogBackend(Protocol):
    async def ps(self) -> list[RunningModel]: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Watchdog:
    def __init__(
        self,
        backend: WatchdogBackend,
        *,
        pinned: Callable[[str], int | None],
        now: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        interval: float = 2.0,
        refresh_keep_alive: Callable[[str], Awaitable[None]] | None = None,
        memory_level: Callable[[], int] = read_memory_level,
    ) -> None:
        self._memory_level = memory_level
        self._expected: dict[str, int] = {}
        self.interval = interval
        self._backend = backend
        self._pinned = pinned
        self._now = now
        self._sleep = sleep
        self._refresh = refresh_keep_alive
        self._listeners: list[Callable[[GuardState], None]] = []
        self._calls: dict[str, int] = {}
        self._speeds: list[float] = []
        self._last_tps: float | None = None
        self._call_alerts: list[tuple[datetime, Alert]] = []
        self._ps_state = GuardState()
        self.state = GuardState()

    def set_expected_size(self, model: str, size: int) -> None:
        """Weights size from /api/tags; lets the poller notice a ps entry that undercounts."""
        self._expected[model] = size

    def subscribe(self, listener: Callable[[GuardState], None]) -> None:
        self._listeners.append(listener)

    def record_call(self, model: str, metrics: CallMetrics, estimate: int | None = None) -> None:
        """Per-call checks; wire to ``OllamaClient.on_call``. ``estimate`` arrives with M2's accounting."""
        first = self._calls.get(model, 0) == 0
        self._calls[model] = self._calls.get(model, 0) + 1
        now = self._now()
        for alert in check_call(metrics, estimate=estimate, first_call=first, speeds=self._speeds):
            self._call_alerts.append((now, alert))
        tps = metrics.tokens_per_second
        if tps is not None:
            self._speeds.append(tps)
            self._last_tps = tps
        self._publish()

    async def poll_once(self) -> GuardState:
        now = self._now()
        models = await self._backend.ps()
        pinned = {m.name: p for m in models if (p := self._pinned(m.name)) is not None}
        mem = self._memory_level()
        alerts = check_ps(models, pinned=pinned, now=now) + check_ps_trust(models, expected=self._expected) + check_memory(mem)
        if self._refresh is not None:
            for m in models:
                if (m.expires_at - now).total_seconds() < EVICTION_WINDOW_S:
                    await self._refresh(m.name)
        head = models[0] if models else None
        self._ps_state = GuardState(
            model=head.name if head else None,
            gpu_fraction=head.gpu_fraction if head else None,
            size=head.size if head else None,
            num_ctx=head.context_length if head else None,
            keep_alive_s=(head.expires_at - now).total_seconds() if head else None,
            memory_level=mem,
            alerts=tuple(alerts),
        )
        return self._publish()

    async def run(self) -> None:
        while True:
            await self.poll_once()
            await self._sleep(self.interval)

    def _publish(self) -> GuardState:
        now = self._now()
        self._call_alerts = [(t, a) for t, a in self._call_alerts if now - t < CALL_ALERT_TTL]
        base = self._ps_state
        self.state = GuardState(
            model=base.model, gpu_fraction=base.gpu_fraction, size=base.size, num_ctx=base.num_ctx,
            tokens_per_second=self._last_tps, keep_alive_s=base.keep_alive_s, memory_level=base.memory_level,
            alerts=base.alerts + tuple(a for _, a in self._call_alerts),
        )
        for listener in self._listeners:
            listener(self.state)
        return self.state

