"""F16 gates 3, 4, 5 and 7 against the stdio fixture (``tests/data/mcp_echo.py``) run as a local subprocess."""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from harness.mcp.config import McpConfig, McpServerConfig
from harness.mcp.hub import McpHub
from harness.mcp.transport import LaunchError, launch_argv, server_env, server_policy
from tests.mcp.helpers import FIXTURE, interpreter_paths, stdio_hub

SECRETS = {"ANTHROPIC_API_KEY": "sk-ant-api03-SECRET", "OPENROUTER_API_KEY": "sk-or-v1-SECRET", "OTHER_API_KEY": "sk-other-SECRET"}


def text(result: Any) -> str:
    return " ".join(str(c.get("text", "")) for c in result["content"])


def problems(notes: list[str]) -> list[str]:
    """The notes about failures: not the standing "runs UNSANDBOXED" line the unsandboxed fixture servers add."""
    return [n for n in notes if "UNSANDBOXED" not in n]


async def call(hub: McpHub, tool: str, args: dict[str, Any] | None = None) -> str:
    return text(await invoke(hub, tool, args))


async def invoke(hub: McpHub, tool: str, args: dict[str, Any] | None = None, server: str = "echo") -> Any:
    return await hub.call(server, tool, {"toolUseId": "1", "name": tool, "input": args or {}}, {})


# --- gate 3 -------------------------------------------------------------------------------------------------------


async def test_the_server_env_is_built_from_scratch_and_home_is_the_state_dir(tmp_path: Path) -> None:
    environ = {**SECRETS, "PATH": "/usr/bin", "DOCS": "docs-value", "UNLISTED": "nope"}
    hub = stdio_hub(tmp_path, environ=environ, env={"DOCS_TOKEN": "${DOCS}"}, pass_env=["DOCS"])
    await hub.start_all()
    assert hub.servers["echo"].status == "connected", hub.notes
    seen = json.loads(text(await invoke(hub, "env")))
    assert seen["HOME"] == str((tmp_path / "data" / "mcp" / "echo").resolve())
    assert seen["DOCS_TOKEN"] == "docs-value"
    assert not any(k.endswith("_API_KEY") for k in seen) and "UNLISTED" not in seen
    assert "sk-" not in " ".join(seen.values()) and seen["TERM"] == "dumb"
    hub.close()


def test_a_missing_pass_env_variable_fails_that_server_not_the_app(tmp_path: Path) -> None:
    cfg = McpServerConfig(command=sys.executable, env={"T": "${NOPE}"}, pass_env=["NOPE"])
    with pytest.raises(LaunchError, match="not set"):
        server_env(cfg, server_policy(cfg, tmp_path / "state", tmp_path), {})


# --- gate 4 -------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
async def test_under_seatbelt_the_fixture_cannot_read_the_home_folder_or_open_a_socket(tmp_path: Path) -> None:
    secret = Path.home() / ".zshrc"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    hub = stdio_hub(tmp_path, sandbox=True, paths=[*interpreter_paths(), str(FIXTURE.parent)])
    try:
        await hub.start_all()
        assert hub.servers["echo"].status == "connected", hub.notes
        assert await call(hub, "echo", {"text": "alive"}) == "alive"
        if secret.exists():
            assert "ERROR" in await call(hub, "read_file", {"path": str(secret)})
        assert await call(hub, "connect", {"host": "127.0.0.1", "port": port}) == "ERROR PermissionError"
        assert await call(hub, "write_state", {"name": "ok.txt"}) == "written"  # its own state dir is writable
        assert (tmp_path / "data" / "mcp" / "echo" / "ok.txt").exists()
        hub.close()
        net = stdio_hub(tmp_path, sandbox=True, network=True, paths=[*interpreter_paths(), str(FIXTURE.parent)])
        await net.start_all()
        assert await call(net, "connect", {"host": "127.0.0.1", "port": port}) == "connected"  # network = true
        net.close()
    finally:
        listener.close()


def test_the_seatbelt_argv_wraps_env_i_around_the_command(tmp_path: Path) -> None:
    cfg = McpServerConfig(command=sys.executable, args=["x.py"])
    policy = server_policy(cfg, tmp_path / "state", tmp_path)
    (tmp_path / "state").mkdir()
    argv = launch_argv(cfg, policy, server_env(cfg, policy, {"ANTHROPIC_API_KEY": "sk-ant-x"}))
    if sys.platform == "darwin":
        assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"] and "(deny network*)" in argv[2]
    assert "/usr/bin/env" in argv and "-i" in argv and "sk-ant-x" not in " ".join(argv)
    assert argv[-2:] == [str(Path(sys.executable)), "x.py"] or argv[-2:] == [sys.executable, "x.py"]


# --- gate 5 -------------------------------------------------------------------------------------------------------


async def test_a_command_that_does_not_exist_is_one_note_and_the_other_server_works(tmp_path: Path) -> None:
    notes: list[str] = []
    config = McpConfig(servers={
        "good": McpServerConfig(command=sys.executable, args=[str(FIXTURE)], sandbox=False),
        "bad": McpServerConfig(command="/no/such/server-binary", sandbox=False),
    })
    hub = McpHub(config, data_dir=tmp_path / "data", project=tmp_path, notify=notes.append, environ={})
    await hub.start_all()
    assert [s.status for s in hub.servers.values()] == ["connected", "failed"]
    failed = problems(notes)
    assert len(failed) == 1 and failed[0].startswith("mcp bad: failed to start") and "/mcp restart bad" in failed[0]
    assert text(await invoke(hub, "echo", {"text": "still here"}, server="good")) == "still here"
    hub.close()


async def test_a_server_that_hangs_at_startup_times_out_with_one_note(tmp_path: Path) -> None:
    notes: list[str] = []
    hub = stdio_hub(tmp_path, notes=notes, args=[str(FIXTURE), "--hang-start"], startup_timeout_s=1)
    started = time.monotonic()
    await hub.start_all()
    assert time.monotonic() - started < 15
    assert hub.servers["echo"].status == "failed" and len(problems(notes)) == 1 and "failed to start" in problems(notes)[0]
    hub.close()


async def test_a_call_that_hangs_past_timeout_returns_a_tool_error_and_one_note(tmp_path: Path) -> None:
    notes: list[str] = []
    hub = stdio_hub(tmp_path, notes=notes, timeout_s=1)
    await hub.start_all()
    result = await invoke(hub, "hang")
    assert result["status"] == "error" and "timed out after 1s" in text(result)
    assert len(problems(notes)) == 1 and "hang timed out" in problems(notes)[0]
    assert hub.servers["echo"].status == "connected"  # slow is not dead
    hub.close()


async def test_a_server_killed_mid_call_is_one_note_and_a_tool_error_and_other_calls_still_answer(tmp_path: Path) -> None:
    notes: list[str] = []
    config = McpConfig(servers={
        "echo": McpServerConfig(command=sys.executable, args=[str(FIXTURE)], sandbox=False, timeout_s=10),
        "twin": McpServerConfig(command=sys.executable, args=[str(FIXTURE)], sandbox=False),
    })
    hub = McpHub(config, data_dir=tmp_path / "data", project=tmp_path, notify=notes.append, environ={})
    await hub.start_all()
    result = await invoke(hub, "die")
    assert result["status"] == "error"
    assert hub.servers["echo"].status == "failed" and len(problems(notes)) == 1 and "connection lost" in problems(notes)[0]
    again = await invoke(hub, "echo", {"text": "a"})
    assert again["status"] == "error" and "/mcp restart echo" in text(again) and len(problems(notes)) == 1
    assert text(await invoke(hub, "echo", {"text": "b"}, server="twin")) == "b"  # the other server is unaffected
    state = await hub.restart("echo")  # and a restart brings the first one back
    assert state.status == "connected" and text(await invoke(hub, "echo", {"text": "c"})) == "c"
    hub.close()


# --- gate 7 -------------------------------------------------------------------------------------------------------


async def test_a_servers_stderr_goes_to_its_log_file_never_the_terminal(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    hub = stdio_hub(tmp_path)
    await hub.start_all()
    assert text(await invoke(hub, "echo", {"text": "x"})) == "x"
    hub.close()
    out = capfd.readouterr()  # capfd sees file descriptors 1 and 2, which a child process writes to directly
    assert out.out == "" and out.err == ""
    assert "fixture-stderr-marker" in hub.log_path("echo").read_text()
    assert str(hub.log_path("echo")) in hub.describe()
