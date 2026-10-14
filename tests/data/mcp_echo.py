"""A tiny stdio MCP server for the F16 gates. Run as ``python mcp_echo.py [--hang-start]``.

Tools: echo, env (what the server was started with), read_file, connect (open a socket), write_state, hang, die."""

import json
import os
import socket
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

if "--hang-start" in sys.argv:  # accepts the process but never answers ``initialize``
    time.sleep(3600)

print("fixture-stderr-marker", file=sys.stderr, flush=True)
server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Return the text."""
    return text


@server.tool()
def env() -> str:
    """The environment this server runs in, as JSON."""
    return json.dumps(dict(os.environ))


@server.tool()
def read_file(path: str) -> str:
    """Read a file."""
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"ERROR {type(exc).__name__}"


@server.tool()
def connect(host: str, port: int) -> str:
    """Open a TCP connection."""
    try:
        socket.create_connection((host, port), timeout=2).close()
    except OSError as exc:
        return f"ERROR {type(exc).__name__}"
    return "connected"


@server.tool()
def write_state(name: str) -> str:
    """Write a file in the working state dir (HOME)."""
    try:
        (Path(os.environ["HOME"]) / name).write_text("x")
    except OSError as exc:
        return f"ERROR {type(exc).__name__}"
    return "written"


@server.tool()
def hang() -> str:
    """Never returns."""
    time.sleep(3600)
    return "late"


@server.tool()
def die() -> str:
    """Kill the server mid-call."""
    os._exit(3)


server.run("stdio")
