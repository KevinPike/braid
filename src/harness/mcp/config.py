"""``[mcp]`` and ``[mcp.servers.*]`` (F16). Everything that can put a harness secret in reach of a server is refused when
the file loads (D14): the error names the server and the field, and nothing starts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, field_validator, model_validator

SERVER_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")  # no "_" (the tool-name separator) and no ":" (the prompt command's)
REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
KEY_VALUE = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}")


def is_secret_name(name: str) -> bool:
    """A variable the harness (or a model provider) reads a key from: ``ANTHROPIC_API_KEY``, ``OPENROUTER_API_KEY`` and
    any other ``*_API_KEY``. None of them is ever forwarded to a server."""
    return name.upper().endswith("_API_KEY")


def resolve(value: str, pass_env: list[str], environ: dict[str, str]) -> str:
    """``value`` with each ``${VAR}`` replaced from ``environ``. Only a ``VAR`` listed in ``pass_env`` is read."""
    def lookup(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in pass_env:
            raise ValueError(f"${{{name}}} is not listed in pass_env")
        if name not in environ:
            raise ValueError(f"${{{name}}} is not set in the environment")
        return environ[name]

    return REFERENCE.sub(lookup, value)


class McpServerConfig(BaseModel):
    """One server: a stdio command (``command``, ``args``) or a Streamable HTTP ``url``."""

    command: str | None = None
    args: list[str] = []
    cwd: str | None = None  # relative to the project
    url: str | None = None
    headers: dict[str, str] = {}
    env: dict[str, str] = {}  # values may name ${VAR}s from pass_env
    pass_env: list[str] = []  # the only variables a ${VAR} may read
    allow: list[str] = []  # bare tool names that run without an Approval; the rest are gated
    network: bool = False  # a Sandbox escape granted once, shown on every Approval
    sandbox: bool = True  # false is logged and shown in /mcp
    paths: list[str] = []  # readable in the Sandbox besides the project, relative to the project
    timeout_s: float = 60.0  # one tool call
    startup_timeout_s: int = 30
    enabled: bool = True

    @field_validator("allow")
    @classmethod
    def _allow_never_ungates(cls, names: list[str]) -> list[str]:
        from harness.approval import ALWAYS_GATED  # late: approval imports hooks, which import config

        bad = sorted(set(names) & ALWAYS_GATED)
        if bad:
            raise ValueError(f"allow may not name an always-gated tool: {', '.join(bad)}")
        return names

    @model_validator(mode="after")
    def _shape_and_secrets(self) -> Self:
        if (self.command is None) == (self.url is None):
            raise ValueError("set exactly one of command (stdio) and url (Streamable HTTP)")
        if self.url is not None and (self.args or self.env or self.cwd):
            raise ValueError("args, env and cwd belong to a stdio server, not a url")
        if self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError(f"url must be http:// or https://, got {self.url!r}")
        for name in self.pass_env:
            if is_secret_name(name):
                raise ValueError(f"pass_env may not forward {name}: harness keys never reach a server")
        for name in self.env:
            if is_secret_name(name):
                raise ValueError(f"env may not set {name}: harness keys never reach a server")
        for where, values in (("env", self.env), ("headers", self.headers)):
            for key, value in values.items():
                if KEY_VALUE.search(value):
                    raise ValueError(f"{where}.{key} holds what looks like a key; use ${{VAR}} and pass_env")
                for ref in REFERENCE.findall(value):
                    if is_secret_name(ref):
                        raise ValueError(f"{where}.{key} references ${{{ref}}}: harness keys never reach a server")
                    if ref not in self.pass_env:
                        raise ValueError(f"{where}.{key} references ${{{ref}}}, which is not listed in pass_env")
        return self

    @property
    def stdio(self) -> bool:
        return self.command is not None


class McpConfig(BaseModel):
    """``[mcp]``: every MCP tool call needs an Approval unless the server's ``allow`` names it (ADR 0012)."""

    enabled: bool = True
    max_resource_chars: int = 20_000  # a resource pulled in with /mcp read is clipped to this
    servers: dict[str, McpServerConfig] = {}

    @field_validator("servers", mode="before")
    @classmethod
    def _names(cls, servers: Any) -> Any:
        for name in servers if isinstance(servers, dict) else ():
            if not SERVER_NAME.match(str(name)):
                raise ValueError(f"server name {name!r} must be lowercase letters, digits and '-'")
        return servers


def state_dir(data_dir: Path, name: str) -> Path:
    """The one place a sandboxed server may write: ``<data_dir>/mcp/<name>/`` (``~/.braid/mcp/<name>/``)."""
    return data_dir / "mcp" / name
