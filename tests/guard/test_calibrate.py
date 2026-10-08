from pathlib import Path

import pytest

from harness.guard.budget import kv_bytes_upper_bound
from harness.guard.calibrate import Calibrator, CalibrationUnreliable, gap_report
from tests.guard.fakes import GIB, FakeBackend


async def test_fits_k_from_two_loads_and_caches_by_version_and_digest(tmp_path: Path) -> None:
    backend = FakeBackend(models={"m": (5 * GIB, 40_000.0)})
    cal = Calibrator(backend, tmp_path / "calibration.json")

    first = await cal.calibrate("m", ctx_lo=4096, ctx_hi=16384)
    assert first.kv_per_token == 40_000.0
    assert first.weights == 5 * GIB
    assert [c for _, c in backend.load_log] == [4096, 16384]

    again = await Calibrator(backend, tmp_path / "calibration.json").calibrate("m")
    assert again == first
    assert len(backend.load_log) == 2  # cache hit, no loads


async def test_new_ollama_version_invalidates_cache(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    old = FakeBackend(models={"m": (GIB, 1000.0)}, version="0.35.1")
    await Calibrator(old, path).calibrate("m")
    new = FakeBackend(models={"m": (GIB, 1000.0)}, version="0.36.0")
    await Calibrator(new, path).calibrate("m")
    assert len(new.load_log) == 2


async def test_refuses_to_calibrate_a_spilling_load(tmp_path: Path) -> None:
    backend = FakeBackend(models={"m": (5 * GIB, 40_000.0)}, gpu_bytes=4 * GIB)
    with pytest.raises(RuntimeError, match="spill"):
        await Calibrator(backend, tmp_path / "c.json").calibrate("m")


def test_gap_report_flags_beyond_25_percent() -> None:
    upper = kv_bytes_upper_bound(layers=32, kv_heads=8, head_dim=128, bytes_per_element=2)  # 131072
    ok = gap_report(measured=upper * 0.9, upper_bound=upper)
    assert ok.within_25pct and not ok.explain_needed
    sliding = gap_report(measured=upper * 0.3, upper_bound=upper)
    assert not sliding.within_25pct and sliding.explain_needed
    assert round(sliding.ratio, 2) == 0.3


async def test_refuses_and_does_not_cache_when_ps_undercounts(tmp_path: Path) -> None:
    """ollama#17251: a draft model makes /api/ps size meaningless, so calibrating would cache garbage."""
    path = tmp_path / "c.json"
    backend = FakeBackend(models={"m": (5 * GIB, 40_000.0)}, draft_only_ps=True)
    with pytest.raises(CalibrationUnreliable, match="17251"):
        await Calibrator(backend, path).calibrate("m")
    assert not path.exists()
