"""F18 gates: braid as an MCP server, driven over the SDK's in-memory transport with a faked Workflow model, a fake
Elicitor and (for the cheap gates) a fake runner. No network, no API spend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mcp_types as types
import pytest
from mcp.client import Client
from mcp.client.session import ClientRequestContext

from harness.config import HarnessConfig, McpServerSideConfig, PathsConfig, ToolsConfig, WorkflowConfig
from harness.mcp.driver import Driver, Models, RunRequest
from harness.mcp.server import Answer, ServerDeps, build_server, make_deps
from harness.openrouter import Catalog, parse_catalog
from harness.sandbox import LocalSandbox
from harness.workflow import bench
from harness.workflow.backend import Choice
from harness.workflow.plan import Plan
from harness.workflow.records import read_log
from harness.workflow.runs import RunInfo, RunStore
from tests.workflow.test_api_runs import CostlyModel, CostlySchemas
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, ScriptedModel, edit
from tests.workflow.test_plan_loop import scripted

KEY_ANTHROPIC = "sk-ant-api03-DISTINCTIVE-fake-key-0123456789"  # gitleaks:allow (a fake, to check redaction)
KEY_OPENROUTER = "sk-or-v1-DISTINCTIVE-fake-key-0123456789"  # gitleaks:allow (a fake, to check redaction)
KEY_OTHER = "sk-other-ABCDEFGH12345"  # gitleaks:allow (a fake, to check redaction)
ENV = {"ANTHROPIC_API_KEY": KEY_ANTHROPIC, "OPENROUTER_API_KEY": KEY_OPENROUTER}

TOOLS = {
    "braid_models", "braid_plans_list", "braid_plan_show", "braid_runs_list", "braid_run_status", "braid_bench_report",
    "braid_plan_start", "braid_build_start", "braid_bench_start", "braid_run_answer", "braid_run_cancel",
}
FREE = TOOLS - {"braid_plan_start", "braid_build_start", "braid_bench_start", "braid_run_answer", "braid_run_cancel"}
HAIKU = Choice("anthropic", "claude-haiku-5-5")


def entry(model_id: str, prompt: str) -> dict[str, Any]:
    return {"id": model_id, "context_length": 200000, "pricing": {"prompt": prompt, "completion": "0.000015"},
            "supported_parameters": ["structured_outputs", "response_format"]}


def fake_catalog() -> Catalog:
    return parse_catalog({"data": [entry("z-ai/glm-5.3", "0.000001"), entry("anthropic/claude-sonnet-5-5", "0.000003")]})


def config(tmp_path: Path, repo: Path | None = None, venv: Path | None = None, **server: Any) -> HarnessConfig:
    test_command = f"{venv}/bin/python -m pytest -q --no-header -p no:cacheprovider" if venv else WorkflowConfig().test_command
    return HarnessConfig(
        tools=ToolsConfig(root=repo or tmp_path), paths=PathsConfig(data_dir=tmp_path / "data"),
        workflow=WorkflowConfig(
            workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", plans_dir=tmp_path / "plans",
            checkpoints_db=tmp_path / "workflows.db", max_polish_rounds=0, test_command=test_command,
        ),
        mcp_server=McpServerSideConfig(**server),
    )


@dataclass
class FakeElicitor:
    """Stands in for the client's human."""

    answer: Answer = "yes"
    asked: list[str] = field(default_factory=list)

    async def __call__(self, ctx: Any, message: str) -> Answer:
        self.asked.append(message)
        return self.answer


@dataclass
class FakeRunner:
    started: list[RunRequest] = field(default_factory=list)
    answered: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    async def start(self, request: RunRequest) -> str:
        self.started.append(request)
        return f"run-{len(self.started)}"

    async def answer(self, run_id: str, answer: str) -> dict[str, Any]:
        self.answered.append((run_id, answer))
        return {"run_id": run_id}

    def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True

    @property
    def touched(self) -> bool:
        return bool(self.started or self.answered or self.cancelled)


def scripted_models(model: Callable[[], Any] | None = None) -> Callable[[HarnessConfig, Choice], Awaitable[Models]]:
    async def models(cfg: HarnessConfig, choice: Choice) -> Models:
        return Models(model or (lambda: ScriptedModel([])), {})

    return models


def deps_for(
    cfg: HarnessConfig, tmp_path: Path, *, elicit: Any = None, runner: Any = None, models: Any = None, **over: Any,
) -> ServerDeps:
    async def catalog(refresh: bool) -> Catalog:
        return fake_catalog()

    over.setdefault("sandbox", LocalSandbox())
    deps = make_deps(
        cfg, elicit=elicit or FakeElicitor(), runner=runner, models=models or scripted_models(), catalog=catalog,
        manifest=tmp_path / "katas.toml", environ=ENV, **over,
    )
    deps.close = None  # the tests read the run index after the client has gone
    return deps


async def no_elicitation(context: ClientRequestContext, params: types.ElicitRequestParams) -> types.ElicitResult:
    return types.ElicitResult(action="accept", content={"approve": True})


@asynccontextmanager
async def connected(deps: ServerDeps, *, elicitation: bool = False) -> AsyncIterator[Client]:
    """A client on the server in memory. The run driver is shut down afterwards, so no task outlives the test."""
    callback = no_elicitation if elicitation else None
    try:
        async with Client(build_server(deps), mode="legacy", elicitation_callback=callback) as client:
            yield client
    finally:
        if isinstance(deps.runner, Driver):
            await deps.runner.shutdown()


async def call(client: Client, name: str, **args: Any) -> tuple[bool, str]:
    result = await client.call_tool(name, args)
    return result.is_error, "".join(c.text for c in result.content if isinstance(c, types.TextContent))


async def call_json(client: Client, name: str, **args: Any) -> dict[str, Any]:
    is_error, text = await call(client, name, **args)
    assert not is_error, text
    data: dict[str, Any] = json.loads(text)
    return data


def spend_row(store: RunStore, run_id: str, cost: float, *, state: str = "finished", cap: float = 0.0) -> None:
    store.create(RunInfo(run_id, "plan", "earlier", "p", "claude-haiku-5-5", state=state, origin="mcp", cost_usd=cost, cap_usd=cap))


# --- gate 1: the tool list ---------------------------------------------------------------------------------------------

async def test_tools_list_is_exactly_the_table_with_hints_and_free_tools_never_touch_the_runner(tmp_path: Path) -> None:
    runner = FakeRunner()
    asker = FakeElicitor()
    cfg = config(tmp_path)
    deps = deps_for(cfg, tmp_path, runner=runner, elicit=asker)
    (cfg.workflow.plans_dir).mkdir(parents=True)
    Plan(goal="add mul", behaviors=["mul multiplies"]).save(cfg.workflow.plans_dir / "p1.json")
    async with connected(deps) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == TOOLS
        for name, tool in tools.items():
            hints = tool.annotations
            assert hints is not None
            if name in FREE:
                assert hints.read_only_hint is True
            elif name != "braid_run_cancel":
                assert (hints.destructive_hint, hints.open_world_hint) == (False, True)
        models = (await call(client, "braid_models"))[1]
        assert "z-ai/glm-5.3" in models and KEY_OPENROUTER not in models
        assert (await call_json(client, "braid_plans_list"))["plans"][0]["id"] == "p1"
        assert "mul multiplies" in (await call(client, "braid_plan_show", id="p1"))[1]
        assert (await call_json(client, "braid_runs_list"))["runs"] == []
        assert "empty" in (await call(client, "braid_bench_report"))[1]
        assert (await call(client, "braid_run_status", run_id="nope"))[0]
        assert (await call(client, "braid_plan_show", id="../../etc/passwd"))[0]  # an id is never a path
    assert not runner.touched and asker.asked == []


# --- gate 2: the human's yes, and the clamp -----------------------------------------------------------------------------

async def test_a_declined_elicitation_starts_nothing_and_an_accepted_one_clamps_to_the_per_run_ceiling(tmp_path: Path) -> None:
    runner, asker = FakeRunner(), FakeElicitor("no")
    deps = deps_for(config(tmp_path, max_cost_usd=0.05), tmp_path, runner=runner, elicit=asker)
    async with connected(deps) as client:
        is_error, text = await call(client, "braid_plan_start", goal="add mul", max_cost_usd=2.0)
        assert is_error and "did not approve" in text and runner.started == []
        assert "Spend cap $0.0500" in asker.asked[0] and "add mul" in asker.asked[0] and "claude-haiku-5-5" in asker.asked[0]

        asker.answer = "yes"
        started = await call_json(client, "braid_plan_start", goal="add mul", max_cost_usd=2.0)
        assert started["status"] == "started" and started["run_id"] == "run-1"
        assert (started["cap_usd"], started["requested_usd"], started["clamped_by"]) == (0.05, 2.0, "per_run")
        [request] = runner.started
        assert request.cap_usd == 0.05 and request.config.workflow.max_cost_usd == 0.05  # the Workflow's own Spend cap

        smaller = await call_json(client, "braid_plan_start", goal="again", max_cost_usd=0.02)
        assert (smaller["cap_usd"], smaller["clamped_by"]) == (0.02, None)  # a lower request is kept
        assert (await call(client, "braid_plan_start", goal="x", max_cost_usd=0))[0]  # nothing to spend is no request


async def test_the_daily_ceiling_clamps_a_request_to_what_is_left(tmp_path: Path) -> None:
    runner = FakeRunner()
    deps = deps_for(config(tmp_path, max_cost_usd=0.50, daily_cost_usd=1.0), tmp_path, runner=runner)
    spend_row(deps.store, "earlier", 0.85)
    async with connected(deps) as client:
        started = await call_json(client, "braid_plan_start", goal="g", max_cost_usd=0.40)
    assert (started["cap_usd"], started["clamped_by"]) == (0.15, "daily")
    assert runner.started[0].config.workflow.max_cost_usd == pytest.approx(0.15)


# --- gate 3: an exhausted ceiling never starts a run ---------------------------------------------------------------------

async def test_with_the_daily_ceiling_used_up_a_spending_call_is_refused_before_anyone_is_asked(tmp_path: Path) -> None:
    runner, asker = FakeRunner(), FakeElicitor()
    deps = deps_for(config(tmp_path, daily_cost_usd=1.0), tmp_path, runner=runner, elicit=asker)
    spend_row(deps.store, "earlier", 0.60)
    spend_row(deps.store, "going", 0.01, state="running", cap=0.40)  # a running run counts at its cap
    build_plan(deps)
    (tmp_path / "katas.toml").write_text('[sets.s]\npairs = [{ kata = "a", plan = "p.json", project = "/nowhere" }]\n')
    async with connected(deps) as client:
        for name, args in (("braid_plan_start", {"goal": "g"}), ("braid_build_start", {"plan_id": "p1"}), ("braid_bench_start", {"set": "s", "label": "l"})):
            is_error, text = await call(client, name, **args)
            assert is_error and "daily ceiling is used up" in text, (name, text)
    assert runner.started == [] and asker.asked == []


async def test_yesterdays_spending_does_not_count_today(tmp_path: Path) -> None:
    deps = deps_for(config(tmp_path, daily_cost_usd=1.0), tmp_path, runner=FakeRunner())
    old = RunInfo("old", "plan", "g", "p", "m", state="finished", origin="mcp", cost_usd=5.0)
    deps.store.create(old)
    deps.store.update("old", state="finished")
    deps.store._db.execute("UPDATE runs SET started = ? WHERE run_id = 'old'", [datetime(2020, 1, 1, tzinfo=UTC).isoformat()])
    assert deps.guard.ledger.spent_today() == 0.0


# --- gate 4: a client without elicitation, and the token mode -----------------------------------------------------------

async def test_a_client_without_elicitation_gets_an_error_in_elicit_mode(tmp_path: Path) -> None:
    runner = FakeRunner()
    deps = make_deps(config(tmp_path), runner=runner, models=scripted_models(), environ=ENV)  # the real elicitor
    async with connected(deps) as client:
        is_error, text = await call(client, "braid_plan_start", goal="g")
        assert is_error and "cannot ask its user" in text and 'confirm = "token"' in text
    assert runner.started == []


async def test_the_real_elicitor_asks_the_clients_human_and_a_cancel_starts_nothing(tmp_path: Path) -> None:
    runner = FakeRunner()
    seen: list[str] = []

    async def human(context: ClientRequestContext, params: types.ElicitRequestParams) -> types.ElicitResult:
        seen.append(params.message)
        return types.ElicitResult(action="accept", content={"approve": len(seen) == 1})

    deps = make_deps(config(tmp_path), runner=runner, models=scripted_models(), environ=ENV)
    async with Client(build_server(deps), mode="legacy", elicitation_callback=human) as client:
        assert (await call_json(client, "braid_plan_start", goal="g"))["status"] == "started"
        is_error, text = await call(client, "braid_plan_start", goal="g")  # accepted the form but left approve false
        assert is_error and "did not approve" in text
    assert len(runner.started) == 1 and "Spend cap" in seen[0]


async def test_token_mode_quotes_then_starts_and_refuses_an_expired_or_altered_token(tmp_path: Path) -> None:
    runner = FakeRunner()
    now = [1000.0]
    deps = deps_for(config(tmp_path, confirm="token"), tmp_path, runner=runner, clock=lambda: now[0])
    async with connected(deps) as client:
        quote = await call_json(client, "braid_plan_start", goal="add mul", max_cost_usd=1.0)
        assert quote["status"] == "confirmation_required" and runner.started == []
        assert quote["cap_usd"] == 0.10 and "Spend cap" in quote["quote"] and quote["confirm_token"]

        started = await call_json(client, "braid_plan_start", goal="add mul", max_cost_usd=1.0, confirm_token=quote["confirm_token"])
        assert started["status"] == "started" and runner.started[0].cap_usd == 0.10
        assert (await call(client, "braid_plan_start", goal="add mul", max_cost_usd=1.0, confirm_token=quote["confirm_token"]))[0]  # used

        altered = await call_json(client, "braid_plan_start", goal="add mul", max_cost_usd=1.0)
        is_error, text = await call(client, "braid_plan_start", goal="add DIV", max_cost_usd=1.0, confirm_token=altered["confirm_token"])
        assert is_error and "other arguments" in text

        late = await call_json(client, "braid_plan_start", goal="late")
        now[0] += 301
        is_error, text = await call(client, "braid_plan_start", goal="late", confirm_token=late["confirm_token"])
        assert is_error and "expired" in text
        assert (await call(client, "braid_plan_start", goal="late", confirm_token="made-up"))[0]
    assert len(runner.started) == 1


# --- gate 6: no key anywhere, and the backend allow-list ----------------------------------------------------------------------

async def test_a_backend_not_in_the_allow_list_is_refused_and_no_key_reaches_a_result(tmp_path: Path) -> None:
    runner, asker = FakeRunner(), FakeElicitor()

    async def leaky(cfg: HarnessConfig, choice: Choice) -> Models:
        raise ValueError(f"upstream said: bad key {KEY_ANTHROPIC} / {KEY_OPENROUTER} / {KEY_OTHER}")

    deps = deps_for(config(tmp_path), tmp_path, runner=runner, elicit=asker, models=leaky)
    async with connected(deps) as client:
        is_error, text = await call(client, "braid_plan_start", goal="g", model="z-ai/glm-5.3")
        assert is_error and "openrouter backend is not allowed" in text
        is_error, text = await call(client, "braid_plan_start", goal="g", model="gemma4:e4b", )
        assert is_error and "frozen" in text
        is_error, text = await call(client, "braid_plan_start", goal="g")  # the model factory fails with keys in its message
        assert is_error and "upstream said" in text
        for secret in (KEY_ANTHROPIC, KEY_OPENROUTER, KEY_OTHER, "DISTINCTIVE"):
            assert secret not in text
    assert runner.started == [] and asker.asked == []

    allowed = deps_for(config(tmp_path, backends=["anthropic", "openrouter"]), tmp_path, runner=runner)
    async with connected(allowed) as client:
        started = await call_json(client, "braid_plan_start", goal="g", model="z-ai/glm-5.3")
    assert started["model"] == "z-ai/glm-5.3" and runner.started[0].choice == Choice("openrouter", "z-ai/glm-5.3")


# --- gates 5 and 6: real runs through the Workflow machinery, with a faked model -------------------------------------------

async def settle(client: Client, run_id: str, want: set[str], seen: list[str] | None = None) -> dict[str, Any]:
    """Wait (long-polling braid_run_status, which sends MCP progress) until the run's status is one of ``want``."""

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        if seen is not None and message:
            seen.append(message)

    for _ in range(200):
        result = await client.call_tool("braid_run_status", {"run_id": run_id, "wait_s": 5}, progress_callback=on_progress)
        text = "".join(c.text for c in result.content if isinstance(c, types.TextContent))
        assert not result.is_error, text
        status: dict[str, Any] = json.loads(text)
        if status["status"] in want:
            return status
    raise AssertionError(f"{run_id} never reached {want}: {status}")


async def test_a_plan_run_waits_at_review_resumes_on_answer_with_basis_mcp_client_and_leaks_no_key(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    model = CostlySchemas(scripted().replies, price=0.001)
    cfg = config(tmp_path, fixture_repo, max_cost_usd=0.10)
    deps = deps_for(cfg, tmp_path, models=scripted_models(lambda: model))
    progress: list[str] = []
    async with connected(deps) as client:
        started = await call_json(client, "braid_plan_start", goal="add a mul helper")
        run_id = started["run_id"]
        waiting = await settle(client, run_id, {"waiting"}, progress)
        assert waiting["waiting"]["kind"] == "review" and waiting["waiting"]["answerable_over_mcp"] is True
        assert "braid_run_answer" in waiting["waiting"]["how_to_continue"] and waiting["waiting"]["payload"]["draft"]
        assert waiting["cap_usd"] == 0.10 and 0 < waiting["cost_usd"] < 0.10

        assert (await call(client, "braid_run_answer", run_id="nope", answer="approve"))[0]
        answered = await call_json(client, "braid_run_answer", run_id=run_id, answer="approve")
        assert answered["answered"] == "review"
        done = await settle(client, run_id, {"finished", "failed"}, progress)
        assert done["status"] == "finished" and done["outcome"] == "approved" and "plan saved" in done["report"]
        assert (await call(client, "braid_run_answer", run_id=run_id, answer="approve"))[0]  # nothing waits any more

        [saved] = deps.config.workflow.plans_dir.glob("*.json")
        assert Plan.load(saved).author is not None and Plan.load(saved).author.backend == "anthropic"  # type: ignore[union-attr]
        listing = await call_json(client, "braid_runs_list")
        assert listing["runs"][0]["run_id"] == run_id and listing["runs"][0]["status"] == "finished"
        everything = json.dumps(listing) + json.dumps(done) + json.dumps(waiting) + "\n".join(progress)
    assert progress and not any(k in everything for k in (KEY_ANTHROPIC, KEY_OPENROUTER, "sk-"))
    routes = [r for r in read_log(cfg.workflow.runs_dir / f"{run_id}.jsonl") if r["node"] == "route"]
    [review] = [r for r in routes if r["from"] == "human_review"]
    assert (review["basis"], review["field"], review["value"]) == ("mcp-client", "answer", "approve")
    assert not any(r["basis"] == "human" for r in routes)


async def test_the_spend_cap_stops_a_run_at_the_granted_figure_and_the_run_index_records_it(
    fixture_repo: Path, tmp_path: Path,
) -> None:
    model = CostlySchemas(scripted().replies, price=0.03)
    deps = deps_for(config(tmp_path, fixture_repo, max_cost_usd=0.10), tmp_path, models=scripted_models(lambda: model))
    async with connected(deps) as client:
        started = await call_json(client, "braid_plan_start", goal="add a mul helper", max_cost_usd=5.0)
        assert started["cap_usd"] == 0.10
        done = await settle(client, started["run_id"], {"finished", "failed"})
    assert (done["status"], done["outcome"]) == ("finished", "spend_cap") and "Spend cap" in done["report"]
    row = deps.store.get(started["run_id"])
    assert row is not None and row.origin == "mcp" and row.cap_usd == 0.10 and row.cost_usd > 0.10  # stops at the next node boundary, so a fan-out may pass the cap
    assert not list(deps.config.workflow.plans_dir.glob("*.json"))  # no Plan saved
    # the day's spending now counts what this run cost
    assert deps.guard.ledger.spent_today() == pytest.approx(row.cost_usd)


async def test_a_run_answer_is_refused_when_the_daily_ceiling_was_used_up_while_it_waited(fixture_repo: Path, tmp_path: Path) -> None:
    model = CostlySchemas(scripted().replies, price=0.001)
    deps = deps_for(config(tmp_path, fixture_repo, daily_cost_usd=1.0), tmp_path, models=scripted_models(lambda: model))
    async with connected(deps) as client:
        run_id = (await call_json(client, "braid_plan_start", goal="add a mul helper"))["run_id"]
        await settle(client, run_id, {"waiting"})
        spend_row(deps.store, "elsewhere", 1.0)
        is_error, text = await call(client, "braid_run_answer", run_id=run_id, answer="approve")
        assert is_error and "daily ceiling" in text
        assert (await call_json(client, "braid_run_status", run_id=run_id))["status"] == "waiting"
        assert (await call_json(client, "braid_run_cancel", run_id=run_id))["status"] == "cancelling"
        assert (await settle(client, run_id, {"paused"}))["outcome"] == "cancelled"


def build_plan(deps: ServerDeps) -> None:
    deps.config.workflow.plans_dir.mkdir(parents=True, exist_ok=True)
    Plan(goal="add mul", behaviors=["mul multiplies"]).save(deps.config.workflow.plans_dir / "p1.json")


async def test_a_build_reaching_sign_off_reports_waiting_and_neither_sign_off_nor_apply_is_possible(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    before = (fixture_repo / "src" / "mod.py").read_text()
    deps = deps_for(config(tmp_path, fixture_repo, fixture_venv), tmp_path,
                    models=scripted_models(lambda: CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.001)))
    build_plan(deps)
    async with connected(deps) as client:
        run_id = (await call_json(client, "braid_build_start", plan_id="p1"))["run_id"]
        waiting = await settle(client, run_id, {"waiting"})
        assert waiting["waiting"]["kind"] == "sign_off" and waiting["waiting"]["answerable_over_mcp"] is False
        assert "TUI" in waiting["waiting"]["how_to_continue"] and f"--resume {run_id}" in waiting["waiting"]["how_to_continue"]
        for answer in ("yes", "approve", "abort"):
            is_error, text = await call(client, "braid_run_answer", run_id=run_id, answer=answer)
            assert is_error and "never possible over MCP" in text and "Sign-off" in text
        assert (await call_json(client, "braid_run_status", run_id=run_id))["status"] == "waiting"
    assert deps.store.get(run_id).outcome == "waiting: sign_off"  # type: ignore[union-attr]
    assert (fixture_repo / "src" / "mod.py").read_text() == before  # nothing was applied
    assert Path(deps.store.get(run_id).workspace).is_dir()  # type: ignore[union-attr]


async def test_a_no_hints_build_ends_waiting_at_the_diff_and_the_diff_cannot_be_applied_over_mcp(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    before = (fixture_repo / "src" / "mod.py").read_text()
    deps = deps_for(config(tmp_path, fixture_repo, fixture_venv), tmp_path,
                    models=scripted_models(lambda: CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.001)))
    build_plan(deps)
    async with connected(deps) as client:
        run_id = (await call_json(client, "braid_build_start", plan_id="p1", no_hints=True))["run_id"]
        waiting = await settle(client, run_id, {"waiting"})
        assert waiting["waiting"]["kind"] == "finish" and "diff" in waiting["waiting"]["payload"]
        is_error, text = await call(client, "braid_run_answer", run_id=run_id, answer="yes")
        assert is_error and "applying the diff" in text
    assert (fixture_repo / "src" / "mod.py").read_text() == before


async def test_a_bench_runs_under_the_clamped_cap_into_the_ledger_and_the_report_tool_reads_it(
    fixture_venv: Path, tmp_path: Path,
) -> None:
    from tests.conftest import make_fixture_repo

    repo = make_fixture_repo(tmp_path / "katas" / "alpha")
    plan_path = tmp_path / "plans-b" / "alpha.json"
    plan_path.parent.mkdir(parents=True)
    Plan(goal="alpha mul", behaviors=["mul multiplies"]).save(plan_path)
    (tmp_path / "katas.toml").write_text(f'[sets.two]\npairs = [{{ kata = "alpha", plan = "plans-b/alpha.json", project = "{repo}" }}]\n')
    deps = deps_for(config(tmp_path, repo, fixture_venv, max_cost_usd=0.50), tmp_path,
                    models=scripted_models(lambda: CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.01)))
    async with connected(deps) as client:
        assert (await call(client, "braid_bench_start", set="nope", label="l"))[0]
        started = await call_json(client, "braid_bench_start", set="two", label="mine", n=1, max_cost_usd=9.0)
        assert (started["cap_usd"], started["clamped_by"]) == (0.50, "per_run")
        done = await settle(client, started["run_id"], {"finished", "failed"})
        assert (done["status"], done["outcome"]) == ("finished", "done") and "mine" in done["report"]
        assert [x["label"] for x in bench.read_ledger(deps.config.paths.bench_ledger)] == ["mine"]
        report = (await call(client, "braid_bench_report", label="mine"))[1]
        assert "mine" in report
        assert (await call(client, "braid_bench_report"))[1].startswith("mine")
    spent = sum(x["summary"]["totals"]["cost_usd"] for x in bench.read_ledger(deps.config.paths.bench_ledger))
    assert spent > 0 and deps.store.get(started["run_id"]).cost_usd == pytest.approx(spent)  # type: ignore[union-attr]


def test_an_old_run_index_gains_the_cost_columns(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT, goal TEXT, project TEXT, model TEXT, state TEXT, "
               "outcome TEXT, plan_path TEXT, workspace TEXT, hints INTEGER, started TEXT, updated TEXT)")
    db.execute("INSERT INTO runs VALUES ('r1','plan','g','p','m','finished','','','',1,'2026-01-01','2026-01-01')")
    db.commit()
    db.close()
    store = RunStore(path)
    row = store.get("r1")
    assert row is not None and (row.cost_usd, row.cap_usd, row.origin) == (0.0, 0.0, "")


# --- gate 7: nothing but JSON-RPC on stdout --------------------------------------------------------------------------------

def test_harness_mcp_as_a_subprocess_writes_only_json_rpc_to_stdout(tmp_path: Path) -> None:
    cfg = tmp_path / "harness.toml"
    cfg.write_text(f'[paths]\ndata_dir = "{tmp_path / "data"}"\n[workflow]\ncheckpoints_db = "{tmp_path / "w.db"}"\n'
                   f'plans_dir = "{tmp_path / "plans"}"\n')
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    proc = subprocess.Popen(
        [sys.executable, "-m", "harness.mcp.server", "--config", str(cfg)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=tmp_path, env={**os.environ, **ENV},
    )
    assert proc.stdin is not None and proc.stdout is not None
    parsed: list[dict[str, Any]] = []
    try:
        for message in messages:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        while {p.get("id") for p in parsed} != {1, 2}:  # each line parses as JSON, or the test fails here
            line = proc.stdout.readline()
            assert line, "the server closed stdout before answering"
            parsed.append(json.loads(line))
        proc.stdin.close()
        rest = proc.stdout.read()
        stderr = proc.stderr.read() if proc.stderr else ""
    finally:
        proc.kill()
        proc.wait(timeout=30)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    assert all(p.get("jsonrpc") == "2.0" for p in parsed) and rest.strip() == ""
    listed = next(p for p in parsed if p.get("id") == 2)
    assert {t["name"] for t in listed["result"]["tools"]} == TOOLS
    seen = json.dumps(parsed) + rest + stderr
    assert KEY_ANTHROPIC not in seen and KEY_OPENROUTER not in seen


async def test_the_terminal_state_and_the_final_cost_are_one_write_even_when_progress_writes_are_locked(
    fixture_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader who sees 'finished' sees the run's cost: the closing write carries both, so a locked index cannot leave a stale
    figure behind a finished state (cost-only writes are made to fail here, as they would under contention)."""
    import sqlite3

    model = CostlySchemas(scripted().replies, price=0.03)
    deps = deps_for(config(tmp_path, fixture_repo, max_cost_usd=0.10), tmp_path, models=scripted_models(lambda: model))
    real_update = deps.store.update

    def update(run_id: str, **changes: Any) -> Any:
        if "state" not in changes and "cost_usd" in changes:
            raise sqlite3.OperationalError("database is locked")
        return real_update(run_id, **changes)

    monkeypatch.setattr(deps.store, "update", update)
    async with connected(deps) as client:
        started = await call_json(client, "braid_plan_start", goal="add a mul helper", max_cost_usd=5.0)
        await settle(client, started["run_id"], {"finished", "failed"})
    row = deps.store.get(started["run_id"])
    assert row is not None and row.state == "finished" and row.cost_usd > 0.10
