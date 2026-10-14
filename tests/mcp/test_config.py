"""F16 gates 2 and 3 (config half): what the config refuses when it loads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from harness.config import HarnessConfig, load_config
from harness.mcp.config import McpConfig, McpServerConfig, resolve


def servers(**server: Any) -> dict[str, Any]:
    return {"servers": {"x": {"command": "uvx", **server}}}


def test_shell_in_allow_is_rejected() -> None:
    for name in ("shell", "write_file", "run_skill_script"):
        with pytest.raises(ValidationError, match="always-gated"):
            McpConfig.model_validate(servers(allow=["git_status", name]))


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "BRAVE_API_KEY"])
def test_env_forwarding_a_key_fails_to_load(name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        McpConfig.model_validate(servers(env={"K": f"${{{name}}}"}, pass_env=[name]))
    with pytest.raises(ValidationError, match=name):
        McpConfig.model_validate(servers(env={"K": f"${{{name}}}"}))  # not in pass_env either
    with pytest.raises(ValidationError, match=name):
        McpConfig.model_validate(servers(pass_env=[name]))
    with pytest.raises(ValidationError, match=name):
        McpConfig.model_validate(servers(env={name: "x"}))


def test_headers_follow_the_same_rule() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        McpConfig.model_validate({"servers": {"d": {"url": "https://d.example/mcp", "headers": {"Authorization": "Bearer ${OPENROUTER_API_KEY}"}}}})
    with pytest.raises(ValidationError, match="not listed in pass_env"):
        McpConfig.model_validate({"servers": {"d": {"url": "https://d.example/mcp", "headers": {"A": "${DOCS_TOKEN}"}}}})
    ok = McpConfig.model_validate({"servers": {"d": {"url": "https://d.example/mcp", "headers": {"A": "Bearer ${DOCS_TOKEN}"}, "pass_env": ["DOCS_TOKEN"]}}})
    assert ok.servers["d"].url == "https://d.example/mcp"


def test_a_literal_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="looks like a key"):
        McpConfig.model_validate(servers(env={"TOKEN": "sk-ant-abcdefgh12345"}))


def test_a_harness_toml_with_a_forwarded_key_does_not_load(tmp_path: Path) -> None:
    path = tmp_path / "harness.toml"
    path.write_text('[mcp.servers.bad]\ncommand = "uvx"\nenv = { K = "${ANTHROPIC_API_KEY}" }\n')
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        load_config(path)


def test_shape_rules() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        McpServerConfig()
    with pytest.raises(ValidationError, match="exactly one"):
        McpServerConfig(command="a", url="https://x")
    with pytest.raises(ValidationError, match="stdio"):
        McpServerConfig(url="https://x", args=["a"])
    with pytest.raises(ValidationError, match="server name"):
        McpConfig.model_validate({"servers": {"a_b": {"command": "x"}}})
    assert HarnessConfig().mcp.servers == {}


def test_resolve_reads_only_pass_env() -> None:
    assert resolve("a-${T}", ["T"], {"T": "1", "OTHER": "2"}) == "a-1"
    with pytest.raises(ValueError, match="not set"):
        resolve("${T}", ["T"], {})
