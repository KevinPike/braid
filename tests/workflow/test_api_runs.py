"""F1 gates 4-6: an API-backed run records tokens and cost, stops at the Spend cap, and keeps its key to itself."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from harness.config import HarnessConfig, WorkflowConfig
from harness.sandbox import LocalSandbox, Sandbox, SeatbeltSandbox
from harness.workflow import cli
from harness.workflow.backend import Choice, choose, make_model
from harness.workflow.cli import SPEND_CAP_EXIT, build_parser, run_build, run_plan, with_model
from harness.workflow.model import AnthropicWorkflowModel, ApiCall, GuardedWorkflowModel
from harness.workflow.plan import Assumption, Plan
from harness.workflow.records import BehaviorStats, Recorder, read_log
from harness.workflow.progress import final_report
from tests.workflow.fakes import SchemaModel
from tests.workflow.test_anthropic_model import message
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, ScriptedModel, edit, run_loop, tree
from tests.workflow.test_plan_loop import run_plan as run_plan_loop, scripted

T = TypeVar("T", bound=BaseModel)
KEY = "sk-ant-api03-DISTINCTIVE-fake-key-0123456789"


def cfg(tmp_path: Path, fixture_venv: Path, **over: Any) -> HarnessConfig:
    return HarnessConfig(workflow=WorkflowConfig(
        workspaces_dir=tmp_path / "ws", runs_dir=tmp_path / "runs", plans_dir=tmp_path / "plans",
        test_command=f"{fixture_venv}/bin/python -m pytest -q --no-header -p no:cacheprovider", **over,
    ))


def api_model(replies: list[BaseModel | str]) -> tuple[AnthropicWorkflowModel, list[dict[str, Any]]]:
    """The real AnthropicWorkflowModel on a mock transport; the client reads ANTHROPIC_API_KEY itself."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["x-api-key"] == KEY  # the SDK's own resolution found it
        reply = replies[len(bodies) - 1]
        text = reply if isinstance(reply, str) else reply.model_dump_json()
        return httpx2.Response(200, json=message(text), headers={"request-id": f"req_{len(bodies)}"})

    client = anthropic.AsyncAnthropic(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0)
    return AnthropicWorkflowModel(client, "claude-haiku-5-5"), bodies


@dataclass
class CostlyModel(ScriptedModel):
    """A scripted model that reports an API call per reply, like AnthropicWorkflowModel."""

    price: float = 0.06
    calls: list[ApiCall] = field(default_factory=list)

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        out = await super().complete(system, prompt, schema)
        self.calls.append(ApiCall(100, 20, 0, 0, self.price, f"req_{len(self.calls) + 1}"))
        return out


@dataclass
class CostlySchemas(SchemaModel):
    """The Plan loop's scripted model, also reporting an API call per reply (``calls`` then mixes both kinds)."""

    price: float = 0.03

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        out = await super().complete(system, prompt, schema)
        self.calls.append(cast(Any, ApiCall(100, 20, 0, 0, self.price, f"req_{len(self.calls)}")))
        return out


# --- backend selection -------------------------------------------------------------------------------------------

def test_the_model_id_picks_the_backend_whatever_the_config_says() -> None:
    config = HarnessConfig()
    assert choose(config, model="claude-haiku-5-5").backend == "anthropic"
    assert choose(config, model="gemma4:e4b").backend == "ollama"
    on_api = HarnessConfig(workflow=WorkflowConfig(backend="anthropic"))
    assert choose(on_api, model="gemma4:e4b").backend == "ollama"  # /build --model gemma4:e4b on an API config


def test_without_a_model_id_each_backend_reads_its_own_model() -> None:
    config = HarnessConfig(workflow=WorkflowConfig(backend="ollama", model="gemma4:e2b", anthropic_model="claude-haiku-5-5"))
    assert (choose(config).backend, choose(config).model_id) == ("ollama", "gemma4:e2b")
    assert choose(HarnessConfig(workflow=WorkflowConfig(backend="ollama"))).model_id == "gemma4:e4b"  # [workflow] model, then the Profile's
    assert choose(HarnessConfig()) == Choice("anthropic", "claude-haiku-5-5")  # the default backend (ADR 0007)
    api = choose(config, backend="anthropic")
    assert (api.backend, api.model_id) == ("anthropic", "claude-haiku-5-5")
    on_api = HarnessConfig(workflow=WorkflowConfig(backend="anthropic"))
    assert (choose(on_api).backend, choose(on_api).model_id) == ("anthropic", "claude-haiku-5-5")
    assert choose(on_api, backend="ollama").model_id == "gemma4:e4b"


def test_make_model_builds_the_anthropic_client_only_for_that_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []
    real = anthropic.AsyncAnthropic

    def client(**kw: Any) -> anthropic.AsyncAnthropic:
        built.append("client")
        assert "api_key" not in kw and kw["max_retries"] == 2  # [anthropic] max_retries; the key is the SDK's to find
        return real(api_key="k", **kw)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", client)
    config = HarnessConfig(workflow=WorkflowConfig(effort="low"))
    assert make_model(config, choose(config, model="gemma4:e4b"), None) is None and built == []
    model = make_model(config, choose(config, model="gemma4:e4b"), object())  # type: ignore[arg-type]
    assert isinstance(model, GuardedWorkflowModel) and built == []
    api = make_model(config, choose(config, model="claude-haiku-5-5"), None)
    assert isinstance(api, AnthropicWorkflowModel) and api.effort == "low" and built == ["client"]


def test_both_subcommands_take_backend_and_max_cost() -> None:
    for argv in (["plan", "goal"], ["build", "--plan", "p.json"]):
        assert build_parser().parse_args(argv).backend is None
        args = build_parser().parse_args([*argv, "--backend", "anthropic", "--max-cost", "0.25"])
        assert args.backend == "anthropic" and args.max_cost == 0.25
        with pytest.raises(SystemExit):
            build_parser().parse_args([*argv, "--backend", "openai"])


async def test_the_anthropic_backend_skips_preflight_and_never_makes_an_ollama_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("an API run must not touch Ollama")

    monkeypatch.setattr(cli, "run_preflight", boom)
    monkeypatch.setattr(cli, "OllamaClient", boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    seen: list[tuple[object, str]] = []

    async def body(model: Any, model_id: str) -> int:
        seen.append((model, model_id))
        return 0

    args = argparse.Namespace(model=None, backend="anthropic", unload=True)
    assert await with_model(args, HarnessConfig(), body) == 0
    [(model, model_id)] = seen
    assert isinstance(model, AnthropicWorkflowModel) and model_id == "claude-haiku-5-5"


# --- gate 4: records carry usage and no key ----------------------------------------------------------------------

def files_under(*roots: Path) -> list[Path]:
    skip = {".git", ".venv"}
    return [p for root in roots for p in root.rglob("*") if p.is_file() and not skip & set(p.relative_to(root).parts)]


async def test_the_run_log_and_summary_carry_tokens_and_cost_and_no_api_key(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    model, _ = api_model([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    out, err = io.StringIO(), io.StringIO()
    code = await run_build(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=model, model_id="claude-haiku-5-5",
        sandbox=LocalSandbox(), config=cfg(tmp_path, fixture_venv), hints=False, ask=lambda _p: "yes", out=out, err=err,
    )
    assert code == 0 and "status: applied" in out.getvalue()
    [log] = (tmp_path / "runs").glob("*.jsonl")
    records = read_log(log)
    modelled = [r for r in records if r["node"] in ("write_test", "write_impl", "polish")]
    assert len(modelled) == 3
    per_call = (10 * 0.10 + 5 * 0.50) / 1_000_000
    for i, r in enumerate(modelled, 1):
        assert (r["input_tokens"], r["output_tokens"], r["cache_read_input_tokens"]) == (10, 5, 0)
        assert r["cost_usd"] == pytest.approx(per_call) and r["request_id"] == f"req_{i}"
    assert all("cost_usd" not in r for r in records if r["node"] not in ("write_test", "write_impl", "polish"))
    [summary] = (fixture_repo / "runs").glob("*.json")
    totals = json.loads(summary.read_text())["totals"]
    assert (totals["input_tokens"], totals["output_tokens"]) == (30, 15) and totals["cost_usd"] == pytest.approx(3 * per_call)
    assert f"${3 * per_call:.4f}" in out.getvalue()  # the end table shows cost
    for path in files_under(tmp_path, fixture_repo) + [summary]:
        assert KEY not in path.read_text(errors="replace"), path
    assert KEY not in out.getvalue() + err.getvalue()


async def test_a_retried_node_sums_its_calls_and_lists_every_request_id(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    model, bodies = api_model(["not json", MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)])
    await run_build(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=model, model_id="claude-haiku-5-5",
        sandbox=LocalSandbox(), config=cfg(tmp_path, fixture_venv), hints=False, ask=lambda _p: "no", out=io.StringIO(), err=io.StringIO(),
    )
    [log] = (tmp_path / "runs").glob("*.jsonl")
    first = next(r for r in read_log(log) if r["node"] == "write_test")
    assert len(bodies) == 4 and first["input_tokens"] == 20 and first["request_ids"] == ["req_1", "req_2"] and first["request_id"] == "req_2"


async def test_ollama_runs_keep_their_records_unchanged(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, ScriptedModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)]), ["yes"], behaviors=["mul"], assumptions=[], hints=False)
    assert run.rec.cost_usd == 0
    assert not any(k in r for r in run.log.lines() for k in ("cost_usd", "input_tokens", "request_id"))


def test_the_recorder_keeps_its_cost_across_dump_and_load() -> None:
    rec = Recorder()
    rec.add_usage([ApiCall(10, 5, 2, 0, 0.04, "r1"), ApiCall(1, 1, 0, 0, 0.01, "r2")])
    again = Recorder()
    again.load(rec.dump())
    assert again.cost_usd == pytest.approx(0.05) and (again.input_tokens, again.cache_read_input_tokens) == (11, 2)
    old = rec.dump()
    for key in ("cost_usd", "input_tokens", "output_tokens", "cache_read_input_tokens"):
        del old[key]  # a checkpoint from before the API backend
    again.load(old)
    assert again.cost_usd == 0


def test_the_end_table_has_a_cost_column_only_for_api_runs() -> None:
    rec = Recorder()
    rec.behaviors.append(BehaviorStats("mul", green=True, cost_usd=0.0123))
    assert "cost" not in final_report(rec, 1.0)
    rec.cost_usd = 0.0123
    assert "cost" in final_report(rec, 1.0) and "$0.0123" in final_report(rec, 1.0)


# --- gate 5: the Spend cap ---------------------------------------------------------------------------------------

async def test_the_build_cli_exits_nonzero_at_the_spend_cap(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    out = io.StringIO()
    code = await run_build(
        Plan(goal="g", behaviors=["mul"]), fixture_repo, model=CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.06),
        model_id="claude-haiku-5-5", sandbox=LocalSandbox(), config=cfg(tmp_path, fixture_venv), hints=False,
        ask=lambda _p: "yes", out=out, err=io.StringIO(),
    )
    assert code == SPEND_CAP_EXIT and "status: spend_cap" in out.getvalue()


async def test_passing_the_spend_cap_stops_at_the_next_node_boundary_and_applies_nothing(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    before = tree(fixture_repo)
    model = CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.06)
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, [], behaviors=["mul"], assumptions=[], hints=False,
        settings=WorkflowConfig(max_cost_usd=0.10),
    )
    # write_test costs 0.06 (under the cap); write_impl brings it to 0.12, so run_green never runs
    assert run.final["status"] == "spend_cap"
    assert run.rec.trace == ["check_assumptions", "write_test", "run_red", "write_impl", "spend_cap"]
    assert run.rec.cost_usd == pytest.approx(0.12) and len(model.calls) == 2
    assert run.payloads == []  # no approval was asked: not even Sign-off or finish
    assert tree(fixture_repo) == before  # nothing reached the project
    last = run.log.lines()[-1]
    assert last["node"] == "spend_cap" and last["spent"] == pytest.approx(0.12) and last["cap"] == 0.10


async def test_a_run_under_the_cap_is_not_stopped(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = CostlyModel([MUL_TEST, MUL_IMPL, edit("src/mod.py", MUL)], price=0.01)
    run = await run_loop(
        fixture_repo, fixture_venv, tmp_path, model, ["yes"], behaviors=["mul"], assumptions=[], hints=False,
        settings=WorkflowConfig(max_cost_usd=0.10),
    )
    assert run.final["status"] == "applied" and tree(fixture_repo)["src/mod.py"] == MUL


async def test_the_plan_loop_stops_at_the_spend_cap_and_saves_no_plan(fixture_repo: Path, tmp_path: Path) -> None:
    model = CostlySchemas(scripted().replies)
    run = await run_plan_loop(fixture_repo, tmp_path, model, [], WorkflowConfig(max_cost_usd=0.10))
    # research is 2 calls (0.06), draft one more (0.09), the Critiques pass the cap; synthesize (one more call) is the
    # node boundary after them, and the human is never asked
    assert run.state["status"] == "spend_cap" and run.rec.trace[-1] == "spend_cap" and "human_review" not in run.rec.trace
    assert not run.plans.exists() or not list(run.plans.glob("*.json"))
    assert run.payloads == []
    critiques = [r for r in run.log.lines() if r["node"] == "critique"]
    assert len(critiques) == 5 and all(r["cost_usd"] == pytest.approx(0.03) for r in critiques)  # each branch got its own call


async def test_the_plan_cli_reports_the_cap(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    out = io.StringIO()
    code = await run_plan(
        "g", fixture_repo, model=CostlySchemas(scripted().replies, price=0.06), model_id="claude-haiku-5-5", config=cfg(tmp_path, fixture_venv),
        ask=lambda _p: "approve", out=out, err=io.StringIO(),
    )
    assert code == SPEND_CAP_EXIT and "Spend cap" in out.getvalue() and "no Plan saved" in out.getvalue()
    assert not list((tmp_path / "plans").glob("*.json"))


# --- gate 6: model-written code cannot see the key ---------------------------------------------------------------

@pytest.mark.parametrize("sandbox", [
    pytest.param(SeatbeltSandbox(), marks=pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")),
    LocalSandbox(),
])
async def test_a_sandbox_command_in_an_api_build_cannot_see_the_api_key(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: Sandbox,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    model, _ = api_model([MUL_TEST, *[MUL_IMPL] * 6])  # the run may go on past the probe; spare replies for any retry
    plan = Plan(goal="g", behaviors=["mul multiplies"], assumptions=[Assumption(text="env is clean", probe="env")])
    await run_build(
        plan, fixture_repo, model=model, model_id="claude-haiku-5-5", sandbox=sandbox,
        config=cfg(tmp_path, fixture_venv, max_polish_rounds=0), hints=False, ask=lambda _p: "no", out=io.StringIO(), err=io.StringIO(),
    )
    [log] = (tmp_path / "runs").glob("*.jsonl")
    probe = next(r for r in read_log(log) if r["node"] == "check_assumptions")["probes"][0]
    assert probe["exit_code"] == 0 and "PATH=" in probe["output_tail"]  # `env` ran, and printed the scrubbed environment
    assert "ANTHROPIC_API_KEY" not in probe["output_tail"] and KEY not in probe["output_tail"]
