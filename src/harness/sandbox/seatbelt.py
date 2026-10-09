"""Sandbox backends: Seatbelt (``sandbox-exec``, ADR 0005) and an unsandboxed one for tests."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from harness.sandbox.policy import RunResult, SandboxPolicy

DEFAULT_TIMEOUT_S = 120.0
MAX_OUTPUT_BYTES = 64 * 1024
KILL_GRACE_S = 0.5
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
DEFAULT_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _quote(path: Path) -> str:
    """An SBPL string literal for the real path; Seatbelt matches real paths (``/tmp`` is ``/private/tmp``)."""
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_profile(policy: SandboxPolicy) -> str:
    """Render ``policy`` as an SBPL profile; later rules win, so each allow follows the deny it carves out of."""
    def subpaths(paths: Sequence[Path]) -> str:
        return " ".join(f"(subpath {_quote(p)})" for p in paths)

    lines = ["(version 1)", "(allow default)"]
    if not policy.network:
        lines.append("(deny network*)")  # includes unix sockets: the ssh-agent socket is an escape route
    lines.append("(deny file-write*)")
    lines.append(f'(allow file-write* {subpaths(policy.writable)} (literal "/dev/null") (literal "/dev/tty"))')
    if policy.denied_read:
        lines.append(f"(deny file-read* {subpaths(policy.denied_read)})")
        reallowed = " ".join([subpaths([*policy.readable, *policy.writable]), *(f"(literal {_quote(p)})" for p in policy.denied_read)])
        lines.append(f"(allow file-read* {reallowed})")
    lines.append("(allow file-read-metadata)")  # without this Python cannot start
    return "\n".join(lines)


def build_env(policy: SandboxPolicy, env: Mapping[str, str] | None) -> dict[str, str]:
    """An empty environment plus PATH, HOME, LANG, TERM and the caller's ``env``; nothing else leaks in."""
    home = policy.writable[0].resolve() if policy.writable else Path("/var/empty")
    out = {"PATH": DEFAULT_PATH, "HOME": str(home), "LANG": "en_US.UTF-8", "TERM": "dumb"}
    if policy.writable:
        tmp = home / ".tmp"
        tmp.mkdir(exist_ok=True)
        out["TMPDIR"] = str(tmp)
    out.update(env or {})
    return out


def _trim(raw: bytes, already_dropped: int, limit: int) -> str:
    """The last ``limit`` bytes, with a head marker counting everything cut."""
    if len(raw) <= limit and not already_dropped:
        return raw.decode(errors="replace")
    return f"[… trimmed {already_dropped + max(len(raw) - limit, 0)} bytes …]\n" + raw[-limit:].decode(errors="replace")


def _kill_group(pid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


async def _finish(proc: asyncio.subprocess.Process, reader: asyncio.Future[None]) -> None:
    await reader
    await proc.wait()


async def run_process(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float, max_output_bytes: int,
) -> RunResult:
    """Run ``argv`` in its own session; on timeout (or cancellation) SIGTERM then SIGKILL the whole process group."""
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=dict(env), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
    )

    async def kill() -> None:
        if _kill_group(proc.pid, signal.SIGTERM):
            await asyncio.sleep(KILL_GRACE_S)
            _kill_group(proc.pid, signal.SIGKILL)

    # Read into our own buffer: a cancelled communicate() would drop what the command printed before the timeout.
    buffer = bytearray()
    dropped = 0

    async def drain() -> None:
        nonlocal dropped
        assert proc.stdout is not None
        while chunk := await proc.stdout.read(65536):
            buffer.extend(chunk)
            if len(buffer) > 2 * max_output_bytes:  # keep memory bounded; the head is dropped anyway
                dropped += len(buffer) - max_output_bytes
                del buffer[:-max_output_bytes]

    reader = asyncio.ensure_future(drain())
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(_finish(proc, reader)), timeout_s)
    except TimeoutError:
        timed_out = True
        await kill()
        await _finish(proc, reader)
    except asyncio.CancelledError:
        reader.cancel()
        await asyncio.shield(kill())
        raise
    code = proc.returncode if proc.returncode is not None else -1
    return RunResult(code, _trim(bytes(buffer), dropped, max_output_bytes), timed_out, time.monotonic() - start)


class SeatbeltSandbox:
    """Runs ``/bin/sh -c command`` under ``sandbox-exec`` with a profile rendered from the policy (macOS only)."""

    def __init__(self, max_output_bytes: int = MAX_OUTPUT_BYTES) -> None:
        self.max_output_bytes = max_output_bytes

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        full_env = build_env(policy, env)  # before the profile exists, so the .tmp dir is made by the harness
        # The profile lives in the system temp dir, which no policy makes writable, so the command cannot edit it.
        with tempfile.NamedTemporaryFile("w", suffix=".sb") as profile:
            profile.write(render_profile(policy))
            profile.flush()
            return await run_process(
                [SANDBOX_EXEC, "-f", profile.name, "/bin/sh", "-c", command],
                cwd=cwd, env=full_env, timeout_s=timeout_s, max_output_bytes=self.max_output_bytes,
            )


class LocalSandbox:
    """The same command with no enforcement: for graph tests only, never selected by config."""

    def __init__(self, max_output_bytes: int = MAX_OUTPUT_BYTES) -> None:
        self.max_output_bytes = max_output_bytes

    async def run(
        self, command: str, *, cwd: Path, policy: SandboxPolicy, timeout_s: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        return await run_process(
            ["/bin/sh", "-c", command], cwd=cwd, env=build_env(policy, env), timeout_s=timeout_s,
            max_output_bytes=self.max_output_bytes,
        )
