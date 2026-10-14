"""The MCP hub (F16): every MCP lifecycle concern behind a few calls (D10 to D18).

The hub owns the ``MCPClient`` of each configured server and outlives agent rebuilds (``/model``), so a server is started
once per app, not once per agent. A failure is a note and an error result, never an exception into the TUI (D16). On
first connect each tool's definition is hashed into a pin file; a changed definition disables that tool until
``/mcp trust`` (D15). ``requires_approval`` is the one gate rule for MCP tools (D12).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from mcp.client.session import ElicitationFnT
from strands.tools.mcp import MCPAgentTool, MCPClient
from strands.types._events import ToolResultEvent
from strands.types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

from harness.hooks import HarnessHooks, McpServerChanged
from harness.mcp.config import McpConfig, McpServerConfig, resolve, state_dir
from harness.mcp.transport import LaunchError, sandboxed_stdio
from harness.skills.catalog import tokens_of

log = logging.getLogger(__name__)

PREFIX = "mcp__"
MAX_NAME = 64  # the Anthropic API's limit on a tool name
PROBE_TIMEOUT_S = 5.0
_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]+")
Transport = Callable[[], Any]
Status = Literal["connected", "failed", "disabled"]


def redact(text: str) -> str:
    """``text`` without anything shaped like an API key: a server's error text can echo one."""
    return _KEY.sub("sk-…", text)


def tool_name(server: str, tool: str) -> str:
    """``mcp__<server>__<tool>`` (D11). A tool name that is too long or has characters the API rejects is cut and gets a
    6-character hash, so two such names stay apart; ``/mcp`` shows the mapping."""
    base = f"{PREFIX}{server}__"
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", tool)
    if clean == tool and len(base + clean) <= MAX_NAME:
        return base + clean
    digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:6]
    return f"{base}{clean[: MAX_NAME - len(base) - 7]}_{digest}"


def definition_hash(spec: ToolSpec, original: str) -> str:
    """What a pin covers: the tool's name, description and input schema, as the model would see them (D15)."""
    canonical = json.dumps(
        {"name": original, "description": spec["description"], "schema": spec["inputSchema"]["json"]},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class Pins:
    """``~/.braid/mcp-pins.json``: server -> tool -> definition hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        self._data: dict[str, dict[str, str]] = data if isinstance(data, dict) else {}

    def get(self, server: str) -> dict[str, str]:
        return dict(self._data.get(server, {}))

    def set(self, server: str, tool: str, digest: str) -> None:
        self._data.setdefault(server, {})[tool] = digest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)


@dataclass
class ToolInfo:
    name: str  # as the agent sees it
    original: str  # as the server names it
    description: str
    digest: str
    tokens: int  # the schema's weight on every request
    annotations: dict[str, Any]
    changed: bool = False  # the definition no longer matches its pin: disabled until /mcp trust
    tool: AgentTool | None = None


@dataclass(frozen=True)
class PromptInfo:
    server: str
    name: str
    description: str
    arguments: tuple[tuple[str, bool], ...]  # (name, required)

    @property
    def command(self) -> str:
        return f"{self.server}:{self.name}"

    def usage(self) -> str:
        return f"/{self.command} " + " ".join(f"{n}=…" if req else f"[{n}=…]" for n, req in self.arguments)


@dataclass
class ServerState:
    name: str
    cfg: McpServerConfig
    status: Status = "disabled"
    detail: str = ""
    client: MCPClient | None = None
    tools: dict[str, ToolInfo] = field(default_factory=dict)  # by agent-facing name
    prompts: dict[str, PromptInfo] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "stdio" if self.cfg.stdio else "http"


def error_result(tool_use: ToolUse, text: str) -> ToolResult:
    return {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": redact(text)}]}


class McpTool(AgentTool):
    """One server tool for the Strands agent. The spec comes from Strands' ``MCPAgentTool``; the call goes through the
    hub, which owns the client (it changes on restart), enforces the pin and the call timeout, and turns every failure into
    an error result."""

    def __init__(self, hub: McpHub, server: str, inner: MCPAgentTool, name: str) -> None:
        super().__init__()
        self.hub, self.server, self.inner, self._name = hub, server, inner, name

    @property
    def tool_name(self) -> str:
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        return {**self.inner.tool_spec, "name": self._name}

    @property
    def tool_type(self) -> str:
        return "python"

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        yield ToolResultEvent(await self.hub.call(self.server, self.inner.mcp_tool.name, tool_use, invocation_state))


class McpHub:
    """``start_all`` once, ``tools()`` for each agent, ``requires_approval`` for the gate."""

    def __init__(
        self, config: McpConfig, *, data_dir: Path, project: Path, hooks: HarnessHooks | None = None,
        notify: Callable[[str], None] | None = None, transports: Mapping[str, Transport] | None = None,
        environ: Mapping[str, str] | None = None, elicitation_callback: ElicitationFnT | None = None,
    ) -> None:
        self.config = config
        self.data_dir = data_dir
        self.project = project
        self.hooks = hooks
        self.notify = notify
        self.notes: list[str] = []
        self.pins = Pins(data_dir / "mcp-pins.json")
        self._transports = dict(transports or {})
        self._environ = dict(environ if environ is not None else {})
        self._elicit = elicitation_callback  # F17 hooks the Asker in here
        self.servers = {n: ServerState(n, c) for n, c in config.servers.items()}

    # --- lifecycle ---------------------------------------------------------------------------------------------------

    def log_path(self, name: str) -> Path:
        """Where a stdio server's stderr goes. Outside its state dir, so the server cannot edit its own log."""
        return self.data_dir / "mcp-logs" / f"{name}.log"

    def _route_client_logs(self) -> None:
        """Strands logs a failed connect at ERROR with a traceback; with no handler of ours, Python would print that to
        stderr, over the TUI. Send the Strands MCP client's records to a file instead."""
        path = self.log_path("client")
        path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("strands.tools.mcp")
        for old in [h for h in logger.handlers if getattr(h, "braid_mcp", False)]:
            logger.removeHandler(old)
            old.close()
        handler = logging.FileHandler(path)
        handler.braid_mcp = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    def _note(self, text: str) -> None:
        text = redact(text)
        self.notes.append(text)
        log.warning(text)
        if self.notify is not None:
            self.notify(text)

    def _publish(self, st: ServerState, detail: str = "") -> None:
        if self.hooks is not None:
            self.hooks.publish(McpServerChanged(st.name, st.status, detail or st.detail))

    async def start_all(self) -> list[ServerState]:
        """Connect every enabled server at once. A server that fails becomes a note; the rest still start."""
        if not self.config.enabled:
            return []
        self._route_client_logs()
        await asyncio.gather(*(self._connect(st) for st in self.servers.values()))
        return list(self.servers.values())

    async def restart(self, name: str) -> ServerState:
        st = self.servers[name]
        await self._disconnect(st)
        await self._connect(st)
        return st

    async def _disconnect(self, st: ServerState) -> None:
        client, st.client = st.client, None
        if client is not None:
            with contextlib.suppress(Exception):  # a dead server's stop() reports its death; we know
                await asyncio.to_thread(client.stop, None, None, None)

    def close(self) -> None:
        for st in self.servers.values():
            if st.client is not None:
                with contextlib.suppress(Exception):
                    st.client.stop(None, None, None)
                st.client = None

    async def _connect(self, st: ServerState) -> None:
        if not st.cfg.enabled:
            st.status, st.detail = "disabled", "enabled = false"
            return
        if st.cfg.stdio and not st.cfg.sandbox:
            self._note(f"mcp {st.name}: runs UNSANDBOXED (sandbox = false)")
        try:
            client = self._client(st)
            st.client = client
            raw, prompts = await asyncio.to_thread(self._open, client)
        except Exception as exc:
            await self._disconnect(st)
            self._fail(st, f"failed to start: {_why(exc)}")
            return
        st.status, st.detail = "connected", ""
        st.tools, st.prompts = self._vet(st, raw), {p.name: replace(p, server=st.name) for p in prompts}
        self._publish(st)

    def _client(self, st: ServerState) -> MCPClient:
        cfg = st.cfg
        timeout = cfg.startup_timeout_s
        if st.name in self._transports:
            return MCPClient(self._transports[st.name], startup_timeout=timeout, elicitation_callback=self._elicit)
        if cfg.stdio:
            transport = sandboxed_stdio(
                cfg, state=state_dir(self.data_dir, st.name), project=self.project, errlog=self.log_path(st.name),
                environ=self._environ,
            )
            return MCPClient(transport, startup_timeout=timeout, elicitation_callback=self._elicit)
        try:
            headers = {k: resolve(v, cfg.pass_env, self._environ) for k, v in cfg.headers.items()}
        except ValueError as exc:
            raise LaunchError(str(exc)) from None
        return MCPClient(url=cfg.url, headers=headers or None, startup_timeout=timeout, elicitation_callback=self._elicit)

    @staticmethod
    def _open(client: MCPClient) -> tuple[list[MCPAgentTool], list[PromptInfo]]:
        """Runs on a worker thread: start the client (bounded by its startup timeout), list tools and prompts."""
        client.start()
        tools: list[MCPAgentTool] = []
        token: str | None = None
        while True:
            page = client.list_tools_sync(token, prefix="")
            tools.extend(page)
            token = page.pagination_token
            if token is None:
                break
        prompts: list[PromptInfo] = []
        try:  # a server without prompts answers "method not found"
            token = None
            while True:
                listed = client.list_prompts_sync(token)
                prompts.extend(
                    PromptInfo("", p.name, p.description or "", tuple((a.name, bool(a.required)) for a in p.arguments or ()))
                    for p in listed.prompts
                )
                token = getattr(listed, "next_cursor", None) or getattr(listed, "nextCursor", None)
                if token is None:
                    break
        except Exception:
            log.debug("server lists no prompts", exc_info=True)
        return tools, prompts

    def _fail(self, st: ServerState, detail: str) -> None:
        st.status, st.detail = "failed", detail
        self._note(f"mcp {st.name}: {detail} (/mcp restart {st.name})")
        self._publish(st)

    # --- pins (D15) --------------------------------------------------------------------------------------------------

    def _vet(self, st: ServerState, raw: list[MCPAgentTool]) -> dict[str, ToolInfo]:
        """Hash each tool: unseen ones are pinned (first connect), a changed one is disabled with a note."""
        self.pins = Pins(self.pins.path)  # another session may have trusted something since this one started
        pinned, tools, changed = self.pins.get(st.name), {}, []
        for mcp_tool in raw:
            original = mcp_tool.mcp_tool.name
            name = tool_name(st.name, original)
            spec = mcp_tool.tool_spec
            digest = definition_hash(spec, original)
            info = ToolInfo(
                name, original, spec["description"], digest, tokens_of(len(json.dumps(spec, default=str))),
                dict(spec.get("annotations") or {}),
            )
            if original not in pinned:
                self.pins.set(st.name, original, digest)
            elif pinned[original] != digest:
                info.changed = True
                changed.append(original)
            info.tool = McpTool(self, st.name, MCPAgentTool(mcp_tool.mcp_tool, mcp_tool.mcp_client, name_override=name), name)
            tools[name] = info
        self.pins.save()
        if changed:
            self._note(
                f"mcp {st.name}: tool definition changed since it was trusted, disabled: {', '.join(changed)} "
                f"(read it in /mcp, then /mcp trust {st.name})"
            )
        return tools

    def trust(self, name: str) -> list[str]:
        """Pin every tool's current definition and enable the changed ones; returns what was enabled."""
        st = self.servers[name]
        enabled = [t.original for t in st.tools.values() if t.changed]
        for t in st.tools.values():
            self.pins.set(name, t.original, t.digest)
            t.changed = False
        self.pins.save()
        if enabled:
            self._publish(st, f"trusted: {', '.join(enabled)}")
        return enabled

    # --- tools and the gate (D11, D12) -------------------------------------------------------------------------------

    def tools(self) -> list[AgentTool]:
        """The tools of every connected server whose definition is trusted, for an agent to register."""
        return [t.tool for st in self.servers.values() if st.status == "connected"
                for t in st.tools.values() if not t.changed and t.tool is not None]

    def find(self, agent_name: str) -> tuple[ServerState, ToolInfo] | None:
        if not agent_name.startswith(PREFIX):
            return None
        for st in self.servers.values():
            if agent_name in st.tools:
                return st, st.tools[agent_name]
        return None

    def requires_approval(self, agent_name: str) -> bool:
        """Every MCP tool call needs an Approval unless its server's ``allow`` names its bare tool name (D12). A name that
        is not a known MCP tool, but wears the prefix, is gated too."""
        if not agent_name.startswith(PREFIX):
            return False
        found = self.find(agent_name)
        return found is None or found[1].original not in found[0].cfg.allow or found[1].changed

    def annotate(self, agent_name: str, tool_input: Mapping[str, Any]) -> str:
        """Lines for the Approval prompt: the server, how it is contained, and its own hints, which are advice only."""
        found = self.find(agent_name)
        if found is None:
            return ""
        st, info = found
        where = f"{st.kind}, " + ("sandbox: on" if st.cfg.sandbox or not st.cfg.stdio else "UNSANDBOXED")
        lines = [f"mcp server: {st.name} ({where}, network: {'ALLOWED' if st.cfg.network or not st.cfg.stdio else 'off'})",
                 f"tool: {info.original}"]
        if info.annotations:
            hints = ", ".join(f"{k}={v}" for k, v in sorted(info.annotations.items()))
            lines.append(f"the server's own hints (advice, not trusted; they skip no Approval): {hints}")
        return "\n".join(lines)

    # --- calls (D16) -------------------------------------------------------------------------------------------------

    def blocked(self, server: str, original: str) -> str | None:
        st = self.servers[server]
        if st.status != "connected" or st.client is None:
            return f"mcp server {server} is not running ({st.detail or st.status}); /mcp restart {server}"
        for info in st.tools.values():
            if info.original == original and info.changed:
                return f"{info.name} is disabled: its definition changed since it was trusted (/mcp trust {server})"
        return None

    async def call(self, server: str, original: str, tool_use: ToolUse, invocation_state: dict[str, Any]) -> ToolResult:
        reason = self.blocked(server, original)
        if reason is not None:
            return error_result(tool_use, reason)
        st = self.servers[server]
        client = st.client
        assert client is not None
        limit = st.cfg.timeout_s
        try:
            result = await asyncio.wait_for(
                client.call_tool_async(
                    tool_use["toolUseId"], original, tool_use["input"], read_timeout_seconds=timedelta(seconds=limit),
                    cancel_signal=getattr(invocation_state.get("agent"), "_cancel_signal", None),
                ),
                limit,
            )
        except TimeoutError:
            self._note(f"mcp {server}: {original} timed out after {limit:g}s")
            return error_result(tool_use, f"{original} timed out after {limit:g}s; the server may be stuck (/mcp restart {server})")
        except Exception as exc:  # the session is gone: nothing more can be called until a restart
            if st.status == "connected":
                self._fail(st, f"connection lost during {original}: {_why(exc)}")
            return error_result(tool_use, f"mcp server {server} is not running; /mcp restart {server}")
        if result["status"] == "error" and "isError" not in result:  # a protocol or transport fault, not the tool's own error
            if not await self._alive(client):
                if st.status == "connected":
                    self._fail(st, f"connection lost during {original}")
            else:
                self._note(f"mcp {server}: {original} failed")
        text = " ".join(str(c.get("text", "")) for c in result["content"] if isinstance(c, Mapping))
        if redact(text) != text:  # a server echoed a key: it is not passed on
            result["content"] = [{"text": redact(text)}]
        return result

    @staticmethod
    async def _alive(client: MCPClient) -> bool:
        try:
            await asyncio.wait_for(asyncio.to_thread(client.list_tools_sync), PROBE_TIMEOUT_S)
        except Exception:
            return False
        return True

    # --- prompts and resources (D17, D18) ----------------------------------------------------------------------------

    def prompt(self, command: str) -> PromptInfo | None:
        """``server:prompt`` -> its PromptInfo, if that server is connected and has it."""
        server, _, prompt = command.partition(":")
        st = self.servers.get(server)
        if st is None or st.status != "connected" or not prompt:
            return None
        return st.prompts.get(prompt)

    def prompt_commands(self) -> list[PromptInfo]:
        return [p for st in self.servers.values() if st.status == "connected" for p in st.prompts.values()]

    async def render_prompt(self, command: str, args: str) -> str:
        """The user turn a ``/server:prompt key=value ...`` becomes. Raises ``ValueError`` with a usage line."""
        info = self.prompt(command)
        if info is None:
            raise ValueError(f"no MCP prompt {command!r} (/mcp lists them)")
        try:
            pairs = [p.partition("=") for p in shlex.split(args)]
        except ValueError as exc:
            raise ValueError(f"{info.usage()}: {exc}") from None
        values = {k: v for k, eq, v in pairs if eq and k}
        known = {n for n, _ in info.arguments}
        missing = [n for n, req in info.arguments if req and n not in values]
        if missing or set(values) - known or any(not eq for _, eq, _ in pairs):
            raise ValueError(f"usage: {info.usage()}" + (f" (missing: {', '.join(missing)})" if missing else ""))
        st = self.servers[info.server]
        assert st.client is not None
        try:
            result = await asyncio.wait_for(asyncio.to_thread(st.client.get_prompt_sync, info.name, values), st.cfg.timeout_s)
        except Exception as exc:
            raise ValueError(f"mcp {info.server}: prompt {info.name} failed: {_why(exc)}") from None
        parts = []
        for message in result.messages:
            text = content_text(message.content)
            parts.append(text if message.role == "user" else f"[{message.role}] {text}")
        return redact("\n\n".join(parts))

    async def read_resource(self, name: str, uri: str) -> str:
        """A resource's text, clipped to ``max_resource_chars``. Raises ``ValueError`` with the reason."""
        st = self.servers.get(name)
        if st is None or st.status != "connected" or st.client is None:
            raise ValueError(f"mcp server {name!r} is not connected (/mcp lists them)")
        try:
            result = await asyncio.wait_for(asyncio.to_thread(st.client.read_resource_sync, uri), st.cfg.timeout_s)
        except Exception as exc:
            raise ValueError(f"mcp {name}: reading {uri} failed: {_why(exc)}") from None
        parts = [getattr(c, "text", None) or f"[{getattr(c, 'mime_type', None) or 'binary'} content omitted]" for c in result.contents]
        text = redact("\n".join(parts))
        limit = self.config.max_resource_chars
        return text if len(text) <= limit else text[:limit] + f"\n[… {len(text) - limit} more characters]"

    # --- /mcp --------------------------------------------------------------------------------------------------------

    def describe(self) -> str:
        if not self.servers:
            return "no MCP servers configured ([mcp.servers.*] in harness.toml)"
        lines: list[str] = []
        for st in self.servers.values():
            weight = sum(t.tokens for t in st.tools.values())
            contain = "http" if not st.cfg.stdio else ("UNSANDBOXED" if not st.cfg.sandbox else "sandboxed")
            net = ", network" if st.cfg.network and st.cfg.stdio else ""
            head = f"{st.name}  [{st.kind}, {contain}{net}]  {st.status}"
            if st.status == "connected":
                head += f"  {len(st.tools)} tools (~{weight} schema tokens)"
            elif st.detail:
                head += f"  ({st.detail})"
            if st.cfg.stdio and st.status != "disabled":
                head += f"  stderr: {self.log_path(st.name)}"
            lines.append(head)
            for t in st.tools.values():
                flags = []
                if t.name != f"{PREFIX}{st.name}__{t.original}":
                    flags.append(f"shortened from {t.original!r}")
                flags.append("DISABLED: changed, /mcp trust " + st.name if t.changed else "allowed" if t.original in st.cfg.allow else "gated")
                lines.append(f"  {t.name}  [{'; '.join(flags)}]  ~{t.tokens} tokens")
                lines.append(f"    {t.description[:200]}{'…' if len(t.description) > 200 else ''}")
            lines += [f"  {p.usage()}  {p.description[:100]}" for p in st.prompts.values()]
        lines += [f"note: {n}" for n in self.notes[-5:]]
        return "mcp servers:\n" + "\n".join(lines)


def content_text(content: Any) -> str:
    """Text of one prompt message part; other kinds are named, not passed on."""
    kind = getattr(content, "type", "")
    if kind == "text":
        return str(content.text)
    if kind == "resource":
        return str(getattr(content.resource, "text", None) or f"[binary resource {content.resource.uri} omitted]")
    return f"[{kind or 'unknown'} content omitted]"


def _why(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)
