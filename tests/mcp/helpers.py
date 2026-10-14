"""Fixtures for the MCP client gates (F16): an in-memory fake server, the stdio fixture, and a gated agent on a fake API."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from harness.hooks import HarnessHooks
from harness.mcp.config import McpConfig, McpServerConfig
from harness.mcp.hub import McpHub
from harness.sandbox import toolchain_paths

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "mcp_echo.py"


def fake_server(description: str = "Say hi.", *, extra_tool: str | None = None, calls: list[str] | None = None) -> MCPServer:
    """An in-memory server with tools ``echo`` (and, optionally, one with a very long name), a prompt and a resource."""
    server = MCPServer("fake")

    def echo(text: str) -> str:
        if calls is not None:
            calls.append("echo")
        return f"echo:{text}"

    echo.__doc__ = description
    server.add_tool(echo, name="echo", description=description)

    def other(x: int) -> str:
        if calls is not None:
            calls.append("other")
        return f"other:{x}"

    server.add_tool(other, name="other", description="Another tool.")
    if extra_tool:
        server.add_tool(other, name=extra_tool, description="A tool with a long name.")

    @server.prompt()
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Please greet {name} warmly."

    @server.resource("memo://notes")
    def notes() -> str:
        """Notes."""
        return "the notes text"

    return server


def in_memory(server: MCPServer) -> Callable[[], Any]:
    return lambda: InMemoryTransport(server)


def fake_hub(
    tmp_path: Path, server: MCPServer, *, allow: list[str] | None = None, notes: list[str] | None = None,
    hooks: HarnessHooks | None = None, **server_cfg: Any,
) -> McpHub:
    """A hub with one server called ``fake``; its transport is in memory, its config is otherwise as a user would write it."""
    config = McpConfig(servers={"fake": McpServerConfig(command="fake", allow=allow or [], **server_cfg)})
    return McpHub(
        config, data_dir=tmp_path / "data", project=tmp_path, hooks=hooks, transports={"fake": in_memory(server)},
        notify=(notes.append if notes is not None else None), environ={},
    )


def stdio_hub(
    tmp_path: Path, *, environ: Mapping[str, str] | None = None, notes: list[str] | None = None, name: str = "echo",
    **cfg: Any,
) -> McpHub:
    """A hub running the stdio fixture as a local subprocess under the current interpreter."""
    server_cfg: dict[str, Any] = {"command": sys.executable, "args": [str(FIXTURE)], "sandbox": False, **cfg}
    return McpHub(
        McpConfig(servers={name: McpServerConfig(**server_cfg)}), data_dir=tmp_path / "data", project=tmp_path,
        notify=(notes.append if notes is not None else None), environ=environ or {},
    )


def interpreter_paths() -> list[str]:
    """What the fixture needs readable under Seatbelt: this venv and the interpreter's prefix (both under ``~``)."""
    return [str(p) for p in toolchain_paths(Path(sys.prefix))]
