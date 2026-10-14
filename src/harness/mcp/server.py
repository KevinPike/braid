"""``harness-mcp``: braid's Workflows as MCP tools over stdio, for a client such as Claude Code (F18, ADR 0013).

The client is an agent, not a human. It may read freely (the model catalog, Plans, runs, the Bench ledger) and may start
bounded spends (``braid_plan_start``, ``braid_build_start``, ``braid_bench_start``) only after a human's yes through MCP
elicitation, or in the weaker ``confirm = "token"`` mode a two-call quote that the client's agent confirms. Two ceilings
(``[mcp_server] max_cost_usd`` and ``daily_cost_usd``) clamp every run's Spend cap, which the Workflow then enforces as it
does for the TUI. It may answer a Plan review, a clarify or Stuck (a Route record with basis ``mcp-client``) but can never
give a Sign-off or apply a diff: those runs stop with status ``waiting`` for a human in the TUI. Nothing but MCP messages
goes to stdout, and no API key reaches a result, an error, a progress message or a log (``scrub``).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import mcp_types as types
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from harness.config import HarnessConfig, load_config
from harness.mcp.driver import HUMAN_ONLY, Driver, Live, Models, Runner, RunnerError, RunBoard, RunRequest, json_safe
from harness.mcp.hub import redact
from harness.mcp.spend import Grant, Refusal, SpendGuard, StoreLedger
from harness.openrouter import Catalog, load_catalog, load_notes, render_models
from harness.sandbox import Sandbox, SeatbeltSandbox
from harness.usage import require_pricing
from harness.workflow import bench
from harness.workflow.backend import Choice, choose, make_model, make_roles, openrouter_ids, prepare_openrouter
from harness.workflow.plan import Plan
from harness.workflow.runs import RunInfo, RunStore

TOKEN_TTL_S = 300.0  # a confirm_token is good for this long, once
MAX_WAIT_S = 120.0  # the longest braid_run_status waits for a change
KEY_VARS = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")

Answer = Literal["yes", "no", "unsupported"]
Elicitor = Callable[[Context[Any, Any], str], Awaitable[Answer]]
ModelFactory = Callable[[HarnessConfig, Choice], Awaitable[Models]]
CatalogLoader = Callable[[bool], Awaitable[Catalog]]

READ_ONLY = types.ToolAnnotations(read_only_hint=True)
SPENDING = types.ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
CONTROL = types.ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True)

CONTINUE = {  # how a client continues from each human pause
    "review": "braid_run_answer: 'approve', 'research: <what to look into>', or notes for a revised draft",
    "clarify": "braid_run_answer: how to proceed, or 'abort'",
    "stuck": "braid_run_answer: a hint for the model, 'skip' or 'abort'",
}


class Approve(BaseModel):
    approve: bool = Field(default=False, description="Start this run and let it spend up to the Spend cap?")


async def ask_human(ctx: Context[Any, Any], message: str) -> Answer:
    """The default ``Elicitor``: a yes/no form for the client's human. "unsupported" when the client cannot elicit."""
    if not ctx.session.check_client_capability(types.ClientCapabilities(elicitation=types.ElicitationCapability())):
        return "unsupported"
    result = await ctx.elicit(message, Approve)
    return "yes" if result.action == "accept" and result.data.approve else "no"


async def prepare_models(config: HarnessConfig, choice: Choice) -> Models:
    """The models a run needs, checked before anything is asked or spent: an OpenRouter model must be in the Catalog with
    structured output (ADR 0009), an Anthropic one must have a price. The keys are read from the environment by the SDK."""
    catalog = None
    if choice.backend == "openrouter":
        catalog, _ = await prepare_openrouter(config, openrouter_ids(config, choice.model_id))
    else:
        require_pricing(choice.model_id)
    roles = make_roles(config, choice.model_id, None, catalog)

    def make() -> Any:
        model = make_model(config, choice, None, catalog)
        assert model is not None
        return model

    return Models(make, roles)


def catalog_loader(config: HarnessConfig) -> CatalogLoader:
    async def load(refresh: bool) -> Catalog:
        return await load_catalog(
            api_key=os.environ.get("OPENROUTER_API_KEY"), cache_path=config.paths.openrouter_models_file,
            max_age_s=0.0 if refresh else 86400.0,
        )

    return load


@dataclass
class ServerDeps:
    """What the server touches, so a test hands it fakes (no model, no network, no human)."""

    config: HarnessConfig
    store: RunStore
    board: RunBoard
    runner: Runner
    guard: SpendGuard
    elicit: Elicitor
    models: ModelFactory
    catalog: CatalogLoader
    manifest: Path = bench.DEFAULT_MANIFEST
    clock: Callable[[], float] = time.monotonic
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    close: Callable[[], Awaitable[None]] | None = None


def make_deps(
    config: HarnessConfig, *, sandbox: Sandbox | None = None, elicit: Elicitor = ask_human, models: ModelFactory = prepare_models,
    catalog: CatalogLoader | None = None, manifest: Path = bench.DEFAULT_MANIFEST, clock: Callable[[], float] = time.monotonic,
    environ: Mapping[str, str] | None = None, store: RunStore | None = None, runner: Runner | None = None,
) -> ServerDeps:
    """The real wiring (``harness-mcp``), with the seams a test replaces."""
    store = store or RunStore(config.workflow.checkpoints_db.expanduser(), timeout=0.05)  # see Driver._retry
    cfg = config.mcp_server
    guard = SpendGuard(cfg.max_cost_usd, cfg.daily_cost_usd, StoreLedger(store))
    board = RunBoard()
    driver = Driver(store, board, guard, sandbox or SeatbeltSandbox(), clock=clock)

    async def close() -> None:
        await driver.shutdown()
        store.close()

    return ServerDeps(
        config, store, board, runner or driver, guard, elicit, models, catalog or catalog_loader(config), manifest, clock,
        os.environ if environ is None else environ, close,
    )


def scrubber(environ: Mapping[str, str]) -> Callable[[str], str]:
    """``text`` without an API key: anything shaped like one, and the exact values of the keys in ``environ``."""
    secrets_ = [v for name in KEY_VARS if len(v := environ.get(name, "")) >= 8]

    def scrub(text: str) -> str:
        for secret in secrets_:
            text = text.replace(secret, "…")
        return redact(text)

    return scrub


class RedactingFilter(logging.Filter):
    """Keeps a key out of the server's log records (stderr), as out of its results."""

    def __init__(self, scrub: Callable[[str], str]) -> None:
        super().__init__()
        self.scrub = scrub

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = self.scrub(record.getMessage()), None
        return True


class TokenBook:
    """The ``confirm = "token"`` mode: a quote is bound to its tool and arguments, good once, for ``TOKEN_TTL_S``."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.issued: dict[str, tuple[str, str, float, float]] = {}

    def issue(self, tool: str, fingerprint: str, cap_usd: float) -> str:
        token = secrets.token_urlsafe(16)
        self.issued[token] = (tool, fingerprint, cap_usd, self.clock() + TOKEN_TTL_S)
        return token

    def redeem(self, token: str, tool: str, fingerprint: str) -> float:
        """The quoted cap, or ToolError when the token is unknown, used, expired or for other arguments."""
        entry = self.issued.pop(token, None)  # single use, even when refused
        if entry is None:
            raise ToolError("unknown or already used confirm_token; call again without it for a new quote")
        quoted_tool, quoted, cap, expires = entry
        if self.clock() > expires:
            raise ToolError("the confirm_token has expired; call again without it for a new quote")
        if (quoted_tool, quoted) != (tool, fingerprint):
            raise ToolError("the confirm_token was issued for other arguments; call again without it for a new quote")
        return cap


def cost_range(lines: list[dict[str, Any]], *, set_name: str | None = None, plan: str | None = None) -> str:
    """What past ledger runs of this set or Plan cost, as ``" Past runs cost $a to $b."``, or ``""``."""
    costs = [
        float(x["summary"].get("totals", {}).get("cost_usd", 0.0)) for x in lines
        if (set_name is not None and x.get("set") == set_name) or (plan is not None and x.get("plan") == plan)
    ]
    return f" Past runs in the Bench ledger cost ${min(costs):.4f} to ${max(costs):.4f} each." if costs else ""


def plan_file(plans_dir: Path, ref: str) -> Path:
    """The saved Plan ``ref`` names: its id (file stem) or a unique prefix of it. Never a path."""
    if not ref or ref != Path(ref).name or ref.startswith("."):
        raise ToolError(f"{ref!r} is not a Plan id (braid_plans_list shows them)")
    exact = plans_dir / f"{ref}.json"
    if exact.is_file():
        return exact
    matches = sorted(plans_dir.glob(f"{glob_escape(ref)}*.json")) if plans_dir.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise ToolError(f"no single Plan matches {ref!r} in {plans_dir} ({len(matches)} do)")


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def build_server(deps: ServerDeps) -> MCPServer[None]:
    cfg, store, board = deps.config, deps.store, deps.board
    scrub = scrubber(deps.environ)
    tokens = TokenBook(deps.clock)
    reserve = asyncio.Lock()
    plans_dir = cfg.workflow.plans_dir.expanduser()

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> Any:
        try:
            yield None
        finally:
            if deps.close is not None:
                await deps.close()

    server: MCPServer[None] = MCPServer(
        "braid", lifespan=lifespan,
        instructions="braid's Workflows. Read tools are free. braid_plan_start, braid_build_start and braid_bench_start spend "
                     "money (a human is asked first) and are capped. Poll braid_run_status; a run waiting for a human says how "
                     "to continue. Sign-off and applying a diff are done by a human in braid's TUI, never here.",
    )

    def tool(name: str, annotations: types.ToolAnnotations) -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
        def register(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
            @functools.wraps(fn)
            async def run(*args: Any, **kwargs: Any) -> str:
                try:
                    return await fn(*args, **kwargs)
                except ToolError as exc:
                    raise ToolError(scrub(str(exc))) from None
                except Exception as exc:  # a failure is an error result, with no key in it
                    raise ToolError(scrub(f"{type(exc).__name__}: {exc}")) from None

            server.add_tool(run, name=name, description=" ".join((fn.__doc__ or "").split()), annotations=annotations, structured_output=False)
            return run

        return register

    def ok(**fields: Any) -> str:
        return scrub(json.dumps(json_safe(fields), indent=2, default=str))

    def info_of(run_id: str) -> RunInfo:
        info = store.resolve(run_id)
        if info is None:
            raise ToolError(f"no single run matches {run_id!r} (braid_runs_list shows them)")
        return info

    def status_of(info: RunInfo, live: Live | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "run_id": info.run_id, "kind": info.kind, "status": info.state, "outcome": info.outcome, "model": info.model,
            "goal": info.goal, "cost_usd": round(info.cost_usd, 6), "cap_usd": info.cap_usd, "started": info.started,
            "owned_by_this_server": live is not None,
        }
        if live is not None:
            out["recent"] = live.lines[-8:]
            if live.waiting is not None:
                kind = live.waiting["kind"]
                human = kind in HUMAN_ONLY
                out["waiting"] = {
                    "kind": kind, "answerable_over_mcp": not human, "payload": live.waiting["payload"],
                    "how_to_continue": (
                        f"A human must do this in braid's TUI (/{info.kind} --resume {info.run_id}); it is not possible over MCP."
                        + (f" The Workspace is {info.workspace}." if info.workspace else "") if human
                        else f"Call {CONTINUE.get(kind, 'braid_run_answer')} (run_id={info.run_id!r})."
                    ),
                }
            if live.report and info.state in ("finished", "failed"):
                out["report"] = live.report
        elif info.state == "waiting":
            out["waiting"] = {"kind": info.outcome.removeprefix("waiting: "), "answerable_over_mcp": False,
                              "how_to_continue": f"Not owned by this server: resume it in braid's TUI (/{info.kind} --resume {info.run_id})."}
        if info.state == "paused":
            out["how_to_continue"] = f"Resume it in braid's TUI (/{info.kind} --resume {info.run_id})."
        return out

    # --- free, read-only tools ---------------------------------------------------------------------------------
    @tool("braid_models", READ_ONLY)
    async def braid_models(refresh: bool = False, pattern: str | None = None, free: bool = False) -> str:
        """The OpenRouter model Catalog with prices, context and structured-output support (Workflows refuse a model without it); also pass claude- ids to anthropic runs. `pattern` is a wildcard such as 'anthropic/*'; `refresh` refetches the free list instead of using the 24 h cache."""
        catalog = await deps.catalog(refresh)
        return scrub(render_models(catalog, free=free, structured_only=True, notes=load_notes(Path("bench/model-notes.toml")), pattern=pattern))

    @tool("braid_plans_list", READ_ONLY)
    async def braid_plans_list() -> str:
        """The saved Plans (newest first), by id, for braid_plan_show and braid_build_start."""
        files = sorted(plans_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if plans_dir.is_dir() else []
        rows = []
        for path in files:
            try:
                plan = Plan.load(path)
            except Exception:  # a stray file in the folder is not a Plan
                continue
            rows.append({"id": path.stem, "goal": plan.goal, "behaviors": len(plan.behaviors),
                         "author": plan.author.model if plan.author else None})
        return ok(plans=rows)

    @tool("braid_plan_show", READ_ONLY)
    async def braid_plan_show(id: str) -> str:
        """One saved Plan as Markdown: its goal, Behaviors, Assumptions and what is out of scope."""
        path = plan_file(plans_dir, id)
        return scrub(f"{path.stem}\n\n{Plan.load(path).to_markdown()}")

    @tool("braid_runs_list", READ_ONLY)
    async def braid_runs_list(limit: int = 20) -> str:
        """Workflow runs, most recent first, from the TUI and from MCP: id, kind, status, model, cost so far and goal."""
        rows = [
            {"run_id": r.run_id, "kind": r.kind, "status": r.state, "outcome": r.outcome, "model": r.model,
             "cost_usd": round(r.cost_usd, 6), "goal": r.goal[:100]}
            for r in store.list(max(1, min(limit, 100)))
        ]
        return ok(runs=rows)

    @tool("braid_run_status", READ_ONLY)
    async def braid_run_status(run_id: str, ctx: Context[Any, Any], wait_s: float = 0.0) -> str:
        """A run's status: running, waiting (for a human; says how to continue), paused, finished or failed, with its cost so far and cap, the latest progress lines and, when finished, its report. `wait_s` (up to 120) holds the call open while the run is running, sending MCP progress notifications, and returns at the first change of status."""
        info = info_of(run_id)
        live = board.get(info.run_id)
        deadline = deps.clock() + max(0.0, min(wait_s, MAX_WAIT_S))
        while live is not None and not live.done and store.get(info.run_id) is not None:
            current = store.get(info.run_id)
            assert current is not None
            if current.state != "running" or deps.clock() >= deadline:
                break
            await ctx.report_progress(len(live.lines), None, scrub(live.lines[-1]) if live.lines else "running")
            try:
                await asyncio.wait_for(live.tick.wait(), max(0.01, deadline - deps.clock()))
            except TimeoutError:
                break
        fresh = info_of(info.run_id)
        return ok(**status_of(fresh, live))

    @tool("braid_bench_report", READ_ONLY)
    async def braid_bench_report(label: str | None = None, compare: str | None = None, frontier: bool = False) -> str:
        """The Bench ledger as tables: no label lists the labels; a label shows its runs; `compare` adds a second label to compare against (a difference within the repeats' range is called noise); `frontier` shows hidden pass rate against cost for the label(s)."""
        lines = bench.read_ledger(cfg.paths.bench_ledger)
        if label is None:
            rows = bench.labels(lines)
            return scrub("\n".join(f"{n}  {runs} runs  from {at}" for n, runs, at in rows) if rows else "the ledger is empty")
        if frontier:
            return scrub(bench.frontier_table(lines, [label, *([compare] if compare else [])]))
        if compare:
            return scrub(bench.compare_table(lines, label, compare))
        return scrub(bench.label_table(lines, label))

    # --- spending tools ----------------------------------------------------------------------------------------
    def refuse_backend(choice: Choice) -> None:
        if choice.backend == "ollama":
            raise ToolError("the Ollama path is frozen (ADR 0007) and is not offered over MCP")
        if choice.backend not in cfg.mcp_server.backends:
            raise ToolError(f"the {choice.backend} backend is not allowed over MCP: add it to [mcp_server] backends")

    def pick(model: str | None) -> Choice:
        try:
            choice = choose(cfg, model=model)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        refuse_backend(choice)
        return choice

    def clamp(requested: float, excluding: str | None = None) -> Grant:
        granted = deps.guard.clamp(requested, excluding=excluding)
        if isinstance(granted, Refusal):
            raise ToolError(f"not started: {granted.reason}")
        return granted

    async def spend(
        ctx: Context[Any, Any], *, tool_name: str, kind: Literal["plan", "build", "bench"], summary: str, args: dict[str, Any],
        requested: float | None, choice: Choice, token: str | None, past: str,
        request: Callable[[HarnessConfig, Models, float], RunRequest],
    ) -> str:
        """Everything between a spending call and a started run, in the order that never wastes a human's yes: refuse a
        backend or an exhausted ceiling first, vet the model, ask (or check the token), clamp again, then start."""
        first = clamp(requested if requested is not None else cfg.workflow.max_cost_usd)
        run_cfg = cfg.model_copy(deep=True)
        models = await deps.models(run_cfg, choice)
        fingerprint = json.dumps({"tool": tool_name, **args}, sort_keys=True, default=str)
        quoted = first.cap_usd
        message = (
            f"Start a {kind} run: {summary}\nModel {choice.model_id} on {choice.backend}; {first.describe()}. "
            f"${first.daily_left_usd:.2f} is left of today's ${cfg.mcp_server.daily_cost_usd:.2f} for MCP-started runs.{past}\n"
            "The run stops at that cap, and it cannot apply a diff or give a Sign-off."
        )
        if cfg.mcp_server.confirm == "token":
            if token is None:
                return ok(status="confirmation_required", quote=message, confirm_token=tokens.issue(tool_name, fingerprint, quoted),
                          expires_in_s=TOKEN_TTL_S, **first.fields(),
                          how_to_continue=f"Call {tool_name} again with the same arguments and this confirm_token. "
                                          "This mode relies on the client's agent to confirm; ask the user first.")
            quoted = tokens.redeem(token, tool_name, fingerprint)
        else:
            answer = await deps.elicit(ctx, message)
            if answer == "unsupported":
                raise ToolError("not started: this client cannot ask its user a question (MCP elicitation), which a spending tool needs. "
                                "Set [mcp_server] confirm = \"token\" for a weaker two-call confirmation by the client's agent.")
            if answer != "yes":
                raise ToolError("not started: the human did not approve the spend")
        async with reserve:  # clamp and record the run together, so two calls cannot be granted the same dollars
            second = clamp(requested if requested is not None else cfg.workflow.max_cost_usd)  # the ceilings may have moved while asking
            grant = min(first, second, key=lambda g: g.cap_usd)
            if quoted < grant.cap_usd:  # a token never buys more than its quote
                grant = replace(grant, cap_usd=quoted, clamped_by=grant.clamped_by or "daily")
            run_cfg.workflow.max_cost_usd = grant.cap_usd
            run_id = await deps.runner.start(request(run_cfg, models, grant.cap_usd))
        return ok(status="started", run_id=run_id, kind=kind, model=choice.model_id, **grant.fields(),
                  how_to_continue=f"Poll braid_run_status(run_id={run_id!r}, wait_s=30); stop it with braid_run_cancel.")

    @tool("braid_plan_start", SPENDING)
    async def braid_plan_start(
        goal: str, ctx: Context[Any, Any], model: str | None = None, max_cost_usd: float | None = None, confirm_token: str | None = None,
    ) -> str:
        """Start the Plan loop on a goal: research, Critiques and rounds of review, in the background; returns a run_id. SPENDS MONEY up to a Spend cap (your max_cost_usd, clamped by braid's ceilings) after the user's yes. Ends waiting for a Plan review that braid_run_answer can approve; a Plan changes nothing until a human builds it."""
        choice = pick(model)
        if not goal.strip():
            raise ToolError("goal is empty")
        return await spend(
            ctx, tool_name="braid_plan_start", kind="plan", summary=goal.strip()[:200],
            args={"goal": goal, "model": model, "max_cost_usd": max_cost_usd}, requested=max_cost_usd, choice=choice, token=confirm_token,
            past="", request=lambda c, m, cap: RunRequest("plan", choice, m.make, m.roles, c, cap, goal=goal.strip()),
        )

    @tool("braid_build_start", SPENDING)
    async def braid_build_start(
        plan_id: str, ctx: Context[Any, Any], model: str | None = None, no_hints: bool = False, max_cost_usd: float | None = None,
        confirm_token: str | None = None,
    ) -> str:
        """Start the Execution loop on a saved Plan: each Behavior goes Red then Green in a sandboxed Workspace, in the background; returns a run_id. SPENDS MONEY up to a Spend cap after the user's yes. It never applies the diff: the run ends waiting for a human at Sign-off or at the diff, to finish in braid's TUI. With no_hints=true code answers Stuck and Sign-off, so only the final diff waits."""
        choice = pick(model)
        path = plan_file(plans_dir, plan_id)
        plan = Plan.load(path)
        return await spend(
            ctx, tool_name="braid_build_start", kind="build", summary=f"build the Plan {path.stem!r}: {plan.goal[:160]}",
            args={"plan_id": path.stem, "model": model, "no_hints": no_hints, "max_cost_usd": max_cost_usd},
            requested=max_cost_usd, choice=choice, token=confirm_token,
            past=cost_range(bench.read_ledger(cfg.paths.bench_ledger), plan=str(path)),
            request=lambda c, m, cap: RunRequest("build", choice, m.make, m.roles, c, cap, plan_path=path, hints=not no_hints),
        )

    @tool("braid_bench_start", SPENDING)
    async def braid_bench_start(
        set: str, label: str, ctx: Context[Any, Any], n: int = 1, model: str | None = None, max_cost_usd: float | None = None,
        confirm_token: str | None = None,
    ) -> str:
        """Run a bench set (bench/katas.toml) n times with one model under a ledger label, in the background; returns a run_id. SPENDS MONEY: max_cost_usd (clamped by braid's ceilings) is the cap for the whole bench, which stops cleanly once spent. Read the result with braid_bench_report."""
        choice = pick(model)
        try:
            sets = bench.load_manifest(deps.manifest)
        except (OSError, ValueError) as exc:
            raise ToolError(f"cannot read the bench manifest {deps.manifest}: {exc}") from None
        if set not in sets:
            raise ToolError(f"no set {set!r} in {deps.manifest} (sets: {', '.join(sets)})")
        if not 1 <= n <= 20:
            raise ToolError("n must be between 1 and 20")
        clean = re.sub(r"[^A-Za-z0-9._-]", "_", label.strip())
        if not clean:
            raise ToolError("label is empty")
        chosen = sets[set]
        runs = len(chosen.pairs) * n
        return await spend(
            ctx, tool_name="braid_bench_start", kind="bench", summary=f"bench {set} ({len(chosen.pairs)} pairs x {n} = {runs} runs), label {clean}",
            args={"set": set, "label": clean, "n": n, "model": model, "max_cost_usd": max_cost_usd}, requested=max_cost_usd,
            choice=choice, token=confirm_token, past=cost_range(bench.read_ledger(cfg.paths.bench_ledger), set_name=set),
            request=lambda c, m, cap: RunRequest("bench", choice, m.make, m.roles, c, cap, bench_set=chosen, repeat=n, label=clean),
        )

    @tool("braid_run_answer", SPENDING)
    async def braid_run_answer(run_id: str, answer: str) -> str:
        """Answer a run that is waiting for a Plan review, a clarify or Stuck; the run resumes and may spend again, within what is left of its cap and the daily ceiling. Refuses Sign-off and applying a diff: those are for a human in braid's TUI."""
        info = info_of(run_id)
        try:
            result = await deps.runner.answer(info.run_id, answer)
        except RunnerError as exc:
            raise ToolError(str(exc)) from None
        return ok(**result, how_to_continue=f"Poll braid_run_status(run_id={info.run_id!r}, wait_s=30).")

    @tool("braid_run_cancel", CONTROL)
    async def braid_run_cancel(run_id: str) -> str:
        """Cancel a run this server started. It stops where it is and can be resumed from its last Checkpoint in braid's TUI; nothing is applied."""
        info = info_of(run_id)
        if not deps.runner.cancel(info.run_id):
            raise ToolError(f"{info.run_id} is not running in this server (its status is {info.state})")
        return ok(run_id=info.run_id, status="cancelling")

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harness-mcp", description="braid's Workflows as MCP tools over stdio.")
    parser.add_argument("--config", type=Path, help="path to harness.toml (default ./harness.toml)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    deps = make_deps(config)
    server = build_server(deps)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr; stdout is the protocol's
    scrub = RedactingFilter(scrubber(deps.environ))
    for logger in (logging.getLogger(), logging.getLogger("mcp")):
        for handler in logger.handlers:
            handler.addFilter(scrub)
    server.run("stdio")


if __name__ == "__main__":
    main()
