"""How a stdio server is launched (F16, D13, D14): under Seatbelt, with an environment built from scratch, and with its
stderr in a log file. The Sandbox interface runs a command to completion; a server is a long-lived pipe, so this builds the
same profile and environment (``render_profile``, ``build_env``) around ``mcp``'s own ``stdio_client``."""

from __future__ import annotations

import contextlib
import shutil
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import anyio
from mcp.client.stdio import StdioServerParameters, stdio_client

from harness.mcp.config import McpServerConfig, resolve
from harness.sandbox import SandboxPolicy
from harness.sandbox.seatbelt import SANDBOX_EXEC, build_env, render_profile

ENV = "/usr/bin/env"


class LaunchError(RuntimeError):
    """The server cannot be started as configured; the message is what the user is told."""


def server_policy(cfg: McpServerConfig, state: Path, project: Path) -> SandboxPolicy:
    """Writable: the server's own state dir. Readable: the project and ``paths``. Network off unless ``network = true``."""
    readable = (project, *(project / p for p in cfg.paths))
    return SandboxPolicy(writable=(state,), readable=tuple(Path(p).expanduser() for p in readable), network=cfg.network)


def server_env(cfg: McpServerConfig, policy: SandboxPolicy, environ: Mapping[str, str]) -> dict[str, str]:
    """PATH, HOME (the state dir), LANG, TERM, TMPDIR and the configured ``env``: nothing is inherited from ``environ``
    except the ``pass_env`` variables a ``${VAR}`` in ``env`` names."""
    try:
        extra = {k: resolve(v, cfg.pass_env, dict(environ)) for k, v in cfg.env.items()}
    except ValueError as exc:
        raise LaunchError(str(exc)) from None
    return build_env(policy, extra)


def launch_argv(cfg: McpServerConfig, policy: SandboxPolicy, env: Mapping[str, str]) -> list[str]:
    """``[sandbox-exec -p profile] env -i K=V ... command args``. ``env -i`` makes ``env`` the child's whole environment,
    which ``stdio_client`` would otherwise merge with a few inherited variables."""
    assert cfg.command is not None
    command = shutil.which(cfg.command, path=env["PATH"]) or shutil.which(cfg.command)
    if command is None:
        raise LaunchError(f"command not found: {cfg.command}")
    argv = [ENV, "-i", *(f"{k}={v}" for k, v in env.items()), command, *cfg.args]
    if not cfg.sandbox:
        return argv
    if sys.platform != "darwin" or not Path(SANDBOX_EXEC).exists():
        raise LaunchError("no Seatbelt on this machine; set sandbox = false for this server to run it unsandboxed")
    return [SANDBOX_EXEC, "-p", render_profile(policy), *argv]


@asynccontextmanager
async def _first_message_by(read: Any, seconds: float) -> AsyncIterator[Any]:
    """``read``, ended early if the server says nothing within ``seconds``. Strands' startup timeout cannot interrupt an
    ``initialize`` that never answers (its ``stop()`` waits for the stuck task), so the stream is closed here instead: the
    session then fails, and leaving the transport kills the server."""
    send, heard = anyio.create_memory_object_stream[Any](0)
    first = anyio.Event()

    async def pump() -> None:
        with contextlib.suppress(anyio.ClosedResourceError, anyio.BrokenResourceError):
            async with send:
                async for message in read:
                    first.set()
                    await send.send(message)

    async def watchdog() -> None:
        with anyio.move_on_after(seconds):
            await first.wait()
            return
        await send.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(pump)
        tg.start_soon(watchdog)
        try:
            yield heard
        finally:
            tg.cancel_scope.cancel()


def sandboxed_stdio(
    cfg: McpServerConfig, *, state: Path, project: Path, errlog: Path, environ: Mapping[str, str],
) -> Callable[[], Any]:
    """A ``transport_callable`` for ``MCPClient``. Called on each (re)connect: it builds the argv then, so a missing
    variable or command fails that connect with a note, and opens ``errlog`` for the server's stderr. Nothing the server
    writes to stderr reaches the terminal, where it would corrupt the TUI."""
    def transport() -> Any:
        state.mkdir(parents=True, exist_ok=True)
        policy = server_policy(cfg, state, project)
        argv = launch_argv(cfg, policy, server_env(cfg, policy, environ))
        cwd = (project / cfg.cwd) if cfg.cwd else project

        @asynccontextmanager
        async def connect() -> AsyncIterator[Any]:
            errlog.parent.mkdir(parents=True, exist_ok=True)
            with errlog.open("a") as log:
                params = StdioServerParameters(command=argv[0], args=argv[1:], cwd=cwd)
                async with stdio_client(params, errlog=log) as (read, write):
                    async with _first_message_by(read, cfg.startup_timeout_s) as heard:
                        yield heard, write

        return cast(Any, connect())

    return transport
