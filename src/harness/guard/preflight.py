"""Preflight: the loud, fail-fast checklist that runs before a model is used."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from harness.guard.backend import Backend
from harness.guard.budget import GIB, kv_bytes_per_token, kv_fixed_bytes, max_ctx
from harness.guard.ollama_client import UnsupportedModelError

MIN_VERSION = (0, 35)
MARGIN = int(1.5 * GIB)
OVERHEAD = int(0.75 * GIB)


class Severity(Enum):
    INFO = "info"
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class Candidate:
    model: str
    num_ctx: int


@dataclass(frozen=True)
class PreflightResult:
    model: str
    num_ctx: int
    budget: int
    checks: tuple[Check, ...]
    weights: int = 0


class PreflightError(RuntimeError):
    def __init__(self, message: str, checks: Sequence[Check]) -> None:
        super().__init__(message)
        self.checks = tuple(checks)


def gpu_limit_bytes(*, wired_limit_mb: int, physical_bytes: int) -> int:
    """Metal limit: ``iogpu.wired_limit_mb`` if set, else a conservative two-thirds of RAM."""
    return wired_limit_mb * 1024 * 1024 if wired_limit_mb > 0 else physical_bytes * 2 // 3


def read_sysctl_int(name: str) -> int:
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def system_gpu_limit() -> int:
    return gpu_limit_bytes(wired_limit_mb=read_sysctl_int("iogpu.wired_limit_mb"), physical_bytes=read_sysctl_int("hw.memsize"))


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("-")[0].split(".")[:2])


async def run_preflight(
    backend: Backend,
    candidates: Sequence[Candidate],
    *,
    env: Mapping[str, str],
    gpu_limit: int,
    helpers: int = 0,
) -> PreflightResult:
    checks: list[Check] = []

    version = await backend.version()
    if _version_tuple(version) == (0, 0):
        # A local build reports 0.0.0, so there is nothing to compare against MIN_VERSION.
        checks.append(Check("version", Severity.WARN, f"Ollama {version} looks like a local build; version check skipped"))
    elif _version_tuple(version) < MIN_VERSION:
        checks.append(Check("version", Severity.FAIL, f"Ollama {version} is older than 0.35"))
        raise PreflightError(f"Ollama {version} is older than 0.35", checks)
    else:
        checks.append(Check("version", Severity.OK, f"Ollama {version}"))

    parallel = env.get("OLLAMA_NUM_PARALLEL", "")
    if parallel == "1":
        checks.append(Check("num_parallel", Severity.OK, "OLLAMA_NUM_PARALLEL=1"))
    else:
        checks.append(Check("num_parallel", Severity.WARN, f"OLLAMA_NUM_PARALLEL={parallel or 'unset'}; parallel slots multiply KV cache"))
    checks.append(Check(
        "flash_attention",
        Severity.OK if env.get("OLLAMA_FLASH_ATTENTION") == "1" else Severity.INFO,
        f"OLLAMA_FLASH_ATTENTION={env.get('OLLAMA_FLASH_ATTENTION', 'unset')}; "
        f"OLLAMA_KV_CACHE_TYPE={env.get('OLLAMA_KV_CACHE_TYPE', 'unset')} (q8_0 roughly halves KV cache)",
    ))

    pulled = {t.name: t for t in await backend.tags()}
    for cand in candidates:
        tag = pulled.get(cand.model)
        if tag is None:
            checks.append(Check("pulled", Severity.WARN, f"{cand.model} is not pulled"))
            continue
        # Budget from architecture, not /api/ps: ps undercounts when a draft model is loaded
        # (ollama#17251). Weights are the on-disk size; calibration is only a cross-check.
        try:
            info = await backend.show(cand.model)
        except UnsupportedModelError as exc:
            checks.append(Check("architecture", Severity.WARN, str(exc)))
            continue
        budget = max_ctx(
            gpu_limit=gpu_limit, margin=MARGIN, helpers=helpers, weights=tag.size,
            overhead=OVERHEAD + kv_fixed_bytes(info), kv_per_token=kv_bytes_per_token(info),
            trained_ctx=info.trained_ctx,
        )
        if cand.num_ctx > budget:
            checks.append(Check("budget", Severity.WARN, f"{cand.model}: num_ctx {cand.num_ctx} exceeds budget {budget}; stepping down"))
            continue
        checks.append(Check("budget", Severity.OK, f"{cand.model}: num_ctx {cand.num_ctx} within budget {budget}"))

        backend.pin(cand.model, cand.num_ctx)
        # Unloading is only needed to change num_ctx; a model already loaded at the pinned value
        # stays put, so the status bar never flashes "no model loaded".
        already = next((m for m in await backend.ps() if m.name == cand.model), None)
        if already is None or already.context_length != cand.num_ctx:
            await backend.unload(cand.model)
        await backend.warm(cand.model)
        loaded = next((m for m in await backend.ps() if m.name == cand.model), None)
        if loaded is None or loaded.spilled:
            await backend.unload(cand.model)
            checks.append(Check("full_gpu", Severity.WARN, f"{cand.model}: spilled or failed to load at num_ctx {cand.num_ctx}"))
            continue
        if loaded.undercounts(expected_size=tag.size):
            checks.append(Check(
                "full_gpu", Severity.WARN,
                f"{cand.model}: /api/ps size {loaded.size / GIB:.1f} GiB is far below {tag.size / GIB:.1f} GiB "
                "(speculative draft model, ollama#17251); cannot confirm the model is fully on GPU",
            ))
        else:
            checks.append(Check("full_gpu", Severity.OK, f"{cand.model}: size_vram == size"))
        return PreflightResult(cand.model, cand.num_ctx, budget, tuple(checks), weights=tag.size)

    detail = "; ".join(c.detail for c in checks if c.severity is Severity.WARN)
    checks.append(Check("fit", Severity.FAIL, "no candidate fits the GPU"))
    raise PreflightError(f"no candidate model fits: {detail}", checks)
