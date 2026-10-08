from datetime import datetime, timedelta, timezone

from harness.guard.ollama_client import CallMetrics, RunningModel
from harness.guard.watchdog import Level, Watchdog, check_call, check_ps
from tests.guard.fakes import GIB, FakeBackend

T0 = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)


def rm(*, size: int = 100, vram: int = 100, ctx: int = 8192, expires_in: float = 1800) -> RunningModel:
    return RunningModel("m", "d", size, vram, T0 + timedelta(seconds=expires_in), ctx)


def metrics(*, prompt: int = 100, evals: int = 50, eval_s: float = 1.0, load_s: float = 0.0) -> CallMetrics:
    return CallMetrics(prompt, evals, eval_s, load_s)


# --- pure checks: one per row of the watchdog table ---

def test_cpu_spill_is_red() -> None:
    (a,) = check_ps([rm(size=100, vram=90)], pinned={"m": 8192}, now=T0)
    assert (a.kind, a.level) == ("spill", Level.RED)


def test_context_length_drift_is_flagged_as_reload() -> None:
    (a,) = check_ps([rm(ctx=4096)], pinned={"m": 8192}, now=T0)
    assert a.kind == "num_ctx_drift" and a.level is Level.RED
    assert "4096" in a.message and "8192" in a.message


def test_healthy_ps_has_no_alerts() -> None:
    assert check_ps([rm()], pinned={"m": 8192}, now=T0) == []


def test_eviction_risk_under_60s_is_yellow() -> None:
    (a,) = check_ps([rm(expires_in=30)], pinned={"m": 8192}, now=T0)
    assert (a.kind, a.level) == ("eviction_risk", Level.YELLOW)


def test_truncation_when_actual_more_than_5pct_below_estimate() -> None:
    assert check_call(metrics(prompt=94), estimate=100, first_call=False, speeds=[])[0].kind == "truncation"
    assert check_call(metrics(prompt=96), estimate=100, first_call=False, speeds=[]) == []
    (a, *_) = check_call(metrics(prompt=50), estimate=100, first_call=False, speeds=[])
    assert a.level is Level.RED and "50" in a.message and "100" in a.message


def test_reload_over_1s_after_first_call_only() -> None:
    assert check_call(metrics(load_s=3.0), estimate=None, first_call=True, speeds=[]) == []
    (a,) = check_call(metrics(load_s=3.0), estimate=None, first_call=False, speeds=[])
    assert (a.kind, a.level) == ("reload", Level.YELLOW)
    assert check_call(metrics(load_s=0.5), estimate=None, first_call=False, speeds=[]) == []


def test_speed_drop_under_half_of_median() -> None:
    history = [50.0, 48.0, 52.0]
    (a,) = check_call(metrics(evals=20, eval_s=1.0), estimate=None, first_call=False, speeds=history)
    assert a.kind == "speed_drop" and a.level is Level.YELLOW
    assert check_call(metrics(evals=40, eval_s=1.0), estimate=None, first_call=False, speeds=history) == []
    assert check_call(metrics(evals=1, eval_s=1.0), estimate=None, first_call=False, speeds=[50.0]) == []  # too few samples


# --- Watchdog: state, poller timing ---

class Clock:
    def __init__(self) -> None:
        self.t = T0

    def now(self) -> datetime:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=s)


async def test_poll_turns_status_red_within_4s_of_a_spill() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (10 * GIB, 0.0)}, gpu_bytes=100 * GIB, clock=clock.now)
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, interval=2.0, memory_level=lambda: 1)
    levels: list[tuple[datetime, Level]] = []
    wd.subscribe(lambda st: levels.append((clock.now(), st.level)))

    spill_at = T0 + timedelta(seconds=1)
    polls = 0
    while clock.now() < T0 + timedelta(seconds=10):
        if clock.now() >= spill_at:
            backend.gpu_bytes = 4 * GIB  # a second large model grabbed the GPU
        await wd.poll_once()
        polls += 1
        await clock.sleep(wd.interval)

    first_red = next(t for t, lvl in levels if lvl is Level.RED)
    assert first_red - spill_at <= timedelta(seconds=4)


async def test_state_reports_gpu_tokens_per_second_and_keep_alive() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (10 * GIB, 0.0)}, clock=clock.now)
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, memory_level=lambda: 1)
    wd.record_call("m", metrics(evals=60, eval_s=2.0))
    st = await wd.poll_once()
    assert st.model == "m" and st.gpu_fraction == 1.0 and st.tokens_per_second == 30.0
    assert 1790 < (st.keep_alive_s or 0) <= 1800
    assert st.level is Level.GREEN


async def test_eviction_risk_refreshes_keep_alive() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (GIB, 0.0)}, clock=clock.now)
    backend.keep_alive_s = 30
    backend.pins["m"] = 8192
    await backend.warm("m")
    refreshed: list[str] = []

    async def refresh(model: str) -> None:
        refreshed.append(model)

    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, refresh_keep_alive=refresh, memory_level=lambda: 1)
    await wd.poll_once()
    assert refreshed == ["m"]


async def test_call_alerts_show_in_state_then_expire() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (GIB, 0.0)}, clock=clock.now)
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, memory_level=lambda: 1)
    wd.record_call("m", metrics(prompt=10), estimate=100)
    assert any(a.kind == "truncation" for a in wd.state.alerts) and wd.state.level is Level.RED
    clock.t += timedelta(seconds=120)
    assert (await wd.poll_once()).level is Level.GREEN


def test_memory_pressure_levels() -> None:
    from harness.guard.watchdog import check_memory

    assert check_memory(1) == []
    assert [(a.kind, a.level) for a in check_memory(2)] == [("memory_pressure", Level.YELLOW)]
    assert [(a.kind, a.level) for a in check_memory(4)] == [("memory_pressure", Level.RED)]


async def test_poll_includes_memory_pressure() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (GIB, 0.0)}, clock=clock.now)
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, memory_level=lambda: 4)
    st = await wd.poll_once()
    assert st.memory_level == 4 and st.level is Level.RED


async def test_ps_undercount_raises_yellow_alert_when_expected_size_known() -> None:
    clock = Clock()
    backend = FakeBackend(models={"m": (10 * GIB, 0.0)}, clock=clock.now, draft_only_ps=True)
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, now=clock.now, sleep=clock.sleep, memory_level=lambda: 1)
    assert (await wd.poll_once()).level is Level.GREEN  # nothing known yet, nothing flagged
    wd.set_expected_size("m", 10 * GIB)
    st = await wd.poll_once()
    assert [(a.kind, a.level) for a in st.alerts] == [("ps_unreliable", Level.YELLOW)]
    assert "17251" in st.alerts[0].message
