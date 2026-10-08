from pathlib import Path

import pytest

from harness.guard.preflight import (
    Candidate,
    PreflightError,
    Severity,
    gpu_limit_bytes,
    run_preflight,
)
from tests.guard.fakes import GIB, FakeBackend

ENV_OK = {"OLLAMA_NUM_PARALLEL": "1"}


def test_gpu_limit_uses_wired_limit_else_two_thirds_of_ram() -> None:
    assert gpu_limit_bytes(wired_limit_mb=20000, physical_bytes=24 * GIB) == 20000 * 1024 * 1024
    assert gpu_limit_bytes(wired_limit_mb=0, physical_bytes=24 * GIB) == 16 * GIB


def run(backend: FakeBackend, tmp_path: Path, cands: list[Candidate], env: dict[str, str] = ENV_OK, gpu: int = 16 * GIB):  # type: ignore[no-untyped-def]
    return run_preflight(backend, cands, env=env, gpu_limit=gpu)


async def test_fitting_request_loads_and_confirms_full_gpu(tmp_path: Path) -> None:
    backend = FakeBackend(models={"small": (3 * GIB, 50_000.0)})
    res = await run(backend, tmp_path, [Candidate("small", 32768)])
    assert (res.model, res.num_ctx) == ("small", 32768)
    assert backend.pins["small"] == 32768 and backend.loaded == {"small": 32768}
    assert not any(c.severity is Severity.FAIL for c in res.checks)


async def test_too_big_num_ctx_steps_down_and_never_loads_a_spilling_model(tmp_path: Path) -> None:
    # big: 14 GiB weights + 1 MiB/token -> budget (16-1.5-14-0.75) < 0 -> refused
    backend = FakeBackend(models={"big": (14 * GIB, 1024.0 * 1024), "small": (3 * GIB, 50_000.0)})
    res = await run(backend, tmp_path, [Candidate("big", 16384), Candidate("small", 16384)])
    assert res.model == "small"
    assert any("big" in c.detail and c.severity is Severity.WARN for c in res.checks)
    assert "big" not in backend.loaded
    ps = await backend.ps()
    assert all(not m.spilled for m in ps)


async def test_refuses_when_nothing_fits(tmp_path: Path) -> None:
    backend = FakeBackend(models={"big": (14 * GIB, 1024.0 * 1024)})
    with pytest.raises(PreflightError, match="big"):
        await run(backend, tmp_path, [Candidate("big", 16384)])
    assert backend.loaded == {}


async def test_old_ollama_fails_loudly(tmp_path: Path) -> None:
    backend = FakeBackend(models={"m": (GIB, 1000.0)}, version="0.34.9")
    with pytest.raises(PreflightError, match="0.35"):
        await run(backend, tmp_path, [Candidate("m", 4096)])


async def test_missing_model_is_skipped_with_failure_detail(tmp_path: Path) -> None:
    backend = FakeBackend(models={"have": (GIB, 1000.0)})
    res = await run(backend, tmp_path, [Candidate("missing", 4096), Candidate("have", 4096)])
    assert res.model == "have"
    assert any(c.name == "pulled" and "missing" in c.detail for c in res.checks)


async def test_parallel_and_kv_cache_env_warnings(tmp_path: Path) -> None:
    backend = FakeBackend(models={"m": (GIB, 1000.0)})
    res = await run(backend, tmp_path, [Candidate("m", 4096)], env={"OLLAMA_NUM_PARALLEL": "4"})
    par = next(c for c in res.checks if c.name == "num_parallel")
    assert par.severity is Severity.WARN
    fa = next(c for c in res.checks if c.name == "flash_attention")
    assert fa.severity is Severity.INFO


async def test_budget_uses_architecture_not_ps(tmp_path: Path) -> None:
    # Weights 3 GiB, k = 50 KB/token: room = 16 - 1.5 - 3 - 0.75 = 10.75 GiB -> far above 32768.
    backend = FakeBackend(models={"m": (3 * GIB, 50_000.0)}, trained_ctx=262144)
    res = await run(backend, tmp_path, [Candidate("m", 16384)])
    assert res.budget == int(10.75 * GIB / 50_000) // 1024 * 1024


async def test_ps_undercount_means_full_gpu_cannot_be_confirmed(tmp_path: Path) -> None:
    backend = FakeBackend(models={"m": (3 * GIB, 50_000.0)}, draft_only_ps=True)
    res = await run(backend, tmp_path, [Candidate("m", 8192)])
    assert res.model == "m"  # still usable
    full = next(c for c in res.checks if c.name == "full_gpu")
    assert full.severity is Severity.WARN and "17251" in full.detail
