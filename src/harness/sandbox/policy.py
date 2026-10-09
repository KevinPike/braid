"""The Sandbox interface: what a process may touch (policy), what came back (result), and who enforces it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SandboxPolicy:
    writable: tuple[Path, ...]  # Workspace (or project root) plus its private tmp dir
    readable: tuple[Path, ...] = ()  # toolchain roots, re-allowed inside denied_read
    denied_read: tuple[Path, ...] = field(default_factory=lambda: (Path.home(),))
    network: bool = False


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    output: str  # stdout+stderr interleaved, tail-trimmed to max_output_bytes
    timed_out: bool
    duration_s: float


class Sandbox(Protocol):
    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = 120.0,
        env: Mapping[str, str] | None = None,
    ) -> RunResult: ...


def toolchain_paths(venv: Path) -> tuple[Path, ...]:
    """What a project's ``.venv`` needs to start Python: the venv and the interpreter's prefix (uv keeps it under ``~``)."""
    python = (venv / "bin" / "python").resolve()
    return (venv.resolve(), python.parent.parent)
