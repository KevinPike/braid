"""OpenRouter as a backend (ADR 0009): the Anthropic model class over its Anthropic-shaped endpoint; MockTransport, no network."""

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

import harness.workflow.cli as cli
from harness.anthropic_api import ApiFailure, make_openrouter_client, openrouter_body
from harness.config import HarnessConfig, OpenRouterAuto, OpenRouterConfig, is_openrouter_id
from harness.openrouter import Catalog, KeyStatus, ModelRefused, parse_catalog
from harness.workflow.backend import choose, openrouter_ids, openrouter_model, prepare_openrouter
from harness.workflow.model import AnthropicWorkflowModel

KEY = "sk-or-v1-secret-key-1234567890"  # gitleaks:allow (a fake, to check redaction)
GOOD = '{"value": 3, "why": "because"}'


class Answer(BaseModel):
    value: int
    why: str


def entry(model_id: str, params: list[str]) -> dict[str, Any]:
    return {"id": model_id, "context_length": 1000, "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "supported_parameters": params}


CATALOG: Catalog = parse_catalog({"data": [
    entry("z-ai/glm-5.3", ["structured_outputs", "tools"]),
    entry("deepseek/deepseek-v4-flash", ["structured_outputs"]),
    entry("old/no-json", ["tools"]),
]})


def message(text: str = GOOD, *, model: str = "z-ai/glm-5.3", cost: float | None = 0.0042) -> dict[str, Any]:
    usage: dict[str, Any] = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1", "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None, "usage": usage,
    }


def make(replies: list[dict[str, Any]], model_id: str = "z-ai/glm-5.3", **kw: Any) -> tuple[AnthropicWorkflowModel, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=replies[len(bodies) - 1], headers={"request-id": f"gen-{len(bodies)}"})

    client = anthropic.AsyncAnthropic(
        api_key=KEY, base_url="https://openrouter.ai/api", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    config = HarnessConfig()
    return openrouter_model(config, model_id, effort=None, catalog=CATALOG, client=client, **kw), bodies


# --- the request ------------------------------------------------------------------------------------------------------------

async def test_the_request_carries_openrouter_routing_and_no_sampling_parameters() -> None:
    model, bodies = make([message()])
    assert (await model.complete("sys", "go", Answer)).value == 3
    body = bodies[0]
    assert body["model"] == "z-ai/glm-5.3"
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_max_price_and_fallbacks_join_the_body() -> None:
    cfg = OpenRouterConfig(max_price={"prompt": 1.0, "completion": 3.0}, fallbacks=["deepseek/deepseek-v4-flash"], data_collection="allow")
    body = openrouter_body(cfg, "z-ai/glm-5.3")
    assert body["provider"] == {"data_collection": "allow", "require_parameters": True, "max_price": {"prompt": 1.0, "completion": 3.0}}
    assert body["models"] == ["z-ai/glm-5.3", "deepseek/deepseek-v4-flash"]
    assert "plugins" not in body


def test_auto_adds_the_router_plugin_and_a_session() -> None:
    cfg = OpenRouterConfig(auto=OpenRouterAuto(cost_tier="low", allowed_models=["anthropic/*"], excluded_models=["anthropic/x"]))
    body = openrouter_body(cfg, "openrouter/auto", session_id="run-1")
    assert body["plugins"] == [{"id": "auto-router", "cost_tier": "low", "allowed_models": ["anthropic/*"], "excluded_models": ["anthropic/x"]}]
    assert body["session_id"] == "run-1"
    assert "session_id" not in openrouter_body(OpenRouterConfig(auto=OpenRouterAuto(session="none")), "openrouter/auto")


# --- cost and the served model ----------------------------------------------------------------------------------------------

async def test_the_reported_cost_is_the_cost_and_no_price_table_entry_is_needed() -> None:
    model, _ = make([message(cost=0.0042)])
    await model.complete("s", "p", Answer)
    [call] = model.calls
    assert call.cost_usd == pytest.approx(0.0042) and model.cost_usd == pytest.approx(0.0042)
    assert call.served_model == "z-ai/glm-5.3" and call.request_id == "gen-1"


async def test_a_free_model_costs_zero() -> None:
    model, _ = make([message(cost=0.0)])
    await model.complete("s", "p", Answer)
    assert model.cost_usd == 0.0


async def test_a_reply_without_cost_fails_loudly_rather_than_counting_zero() -> None:
    model, _ = make([message(cost=None)])
    with pytest.raises(ApiFailure, match="no cost"):
        await model.complete("s", "p", Answer)


async def test_a_served_model_without_structured_output_fails_the_call() -> None:
    model, bodies = make([message(model="old/no-json")], model_id="openrouter/auto")
    with pytest.raises(ApiFailure, match="structured outputs"):
        await model.complete("s", "p", Answer)
    assert len(bodies) == 1  # not a parse retry


async def test_an_auto_run_records_each_served_model_and_passes_an_unknown_one() -> None:
    model, _ = make([message(model="deepseek/deepseek-v4-flash"), message(model="brand/new-model-0929")], model_id="openrouter/auto")
    await model.complete("s", "p", Answer)
    await model.complete("s", "p", Answer)
    assert [c.served_model for c in model.calls] == ["deepseek/deepseek-v4-flash", "brand/new-model-0929"]


# --- choosing and preflight ------------------------------------------------------------------------------------------------

def test_a_vendor_slash_model_id_picks_openrouter() -> None:
    cfg = HarnessConfig()
    assert choose(cfg, model="z-ai/glm-5.3").backend == "openrouter"
    assert choose(cfg, model="openrouter/auto").backend == "openrouter"
    assert choose(cfg, model="claude-haiku-5-5").backend == "anthropic"
    assert choose(cfg, model="hf.co/org/model:q4").backend == "ollama"
    assert choose(cfg, model="gemma4:e4b").backend == "ollama"
    assert not is_openrouter_id("claude-haiku-5-5")


def test_the_openrouter_backend_has_no_default_model() -> None:
    cfg = HarnessConfig()
    with pytest.raises(ValueError, match="--model"):
        choose(cfg, backend="openrouter")
    cfg.workflow.openrouter_model = "z-ai/glm-5.3"
    assert choose(cfg, backend="openrouter").model_id == "z-ai/glm-5.3"


def test_the_ids_a_run_would_call_include_roles_and_the_arbiter() -> None:
    cfg = HarnessConfig()
    cfg.workflow.roles.tester.model = "deepseek/deepseek-v4-flash"
    cfg.workflow.roles.arbiter.model = "claude-sonnet-5-5"
    assert openrouter_ids(cfg, "z-ai/glm-5.3") == ["z-ai/glm-5.3", "deepseek/deepseek-v4-flash"]


@pytest.fixture
def preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_catalog(**_: Any) -> Catalog:
        return CATALOG

    async def fake_key(**_: Any) -> KeyStatus:
        return KeyStatus(5.0, 4.5, 0.5, 1000, 1000)

    monkeypatch.setattr("harness.workflow.backend.load_catalog", fake_catalog)
    monkeypatch.setattr("harness.workflow.backend.key_status", fake_key)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)


async def test_preflight_refuses_a_model_without_structured_output(preflight: None) -> None:
    catalog, key = await prepare_openrouter(HarnessConfig(), ["z-ai/glm-5.3"])
    assert catalog is CATALOG and key.limit_remaining == 4.5
    with pytest.raises(ModelRefused, match="structured outputs"):
        await prepare_openrouter(HarnessConfig(), ["z-ai/glm-5.3", "old/no-json"])


async def test_preflight_for_auto_needs_an_allow_list_of_structured_models(preflight: None) -> None:
    cfg = HarnessConfig()
    with pytest.raises(ModelRefused, match="allowed_models"):
        await prepare_openrouter(cfg, ["openrouter/auto"])
    cfg.openrouter.auto.allowed_models = ["old/*"]
    with pytest.raises(ModelRefused):
        await prepare_openrouter(cfg, ["openrouter/auto"])
    cfg.openrouter.auto.allowed_models = ["z-ai/*", "old/*"]
    await prepare_openrouter(cfg, ["openrouter/auto"])


async def test_preflight_needs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        await prepare_openrouter(HarnessConfig(), ["z-ai/glm-5.3"])
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_openrouter_client(OpenRouterConfig())


async def test_a_refused_model_exits_two_before_any_call(preflight: None, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    ran: list[str] = []

    async def body(_m: Any, model_id: str) -> int:
        ran.append(model_id)
        return 0

    args = argparse.Namespace(model="old/no-json", backend=None, unload=False)
    assert await cli.with_model(args, HarnessConfig(), body) == 2
    assert not ran and "structured outputs" in capsys.readouterr().err


# --- the key ----------------------------------------------------------------------------------------------------------------

def test_a_key_in_an_error_message_is_redacted() -> None:
    failure = ApiFailure("AuthenticationError", f"bad key {KEY} sent")
    assert KEY not in failure.message and "sk-or-…" in failure.message


async def test_the_key_goes_to_the_endpoint_and_nowhere_else(tmp_path: Path) -> None:
    model, bodies = make([message()])
    await model.complete("s", "p", Answer)
    assert KEY not in json.dumps(bodies)


def test_openrouter_is_not_imported_by_the_guard() -> None:
    import ast

    guard = Path(__file__).parents[2] / "src" / "harness" / "guard"
    for path in guard.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any("openrouter" in n for n in names), f"{path.name} imports {names}"


# --- thinking sampling ------------------------------------------------------------------------------------------------------

def _sampled(sample: float, seed: int = 0) -> tuple[list[dict[str, Any]], AnthropicWorkflowModel]:
    import random

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=message(), headers={"request-id": f"gen-{len(bodies)}"})

    client = anthropic.AsyncAnthropic(
        api_key=KEY, base_url="https://openrouter.ai/api", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    model = AnthropicWorkflowModel(client, "z-ai/glm-5.3", reported_cost=True, thinking_sample=sample, rng=random.Random(seed))
    return bodies, model


async def test_a_sampled_run_sends_thinking_on_for_some_calls_and_disabled_for_the_rest() -> None:
    bodies, model = _sampled(0.25)
    for _ in range(40):
        await model.complete("s", "p", Answer)
    on = [c.thinking_on for c in model.calls]
    assert all(isinstance(x, bool) for x in on) and 0 < sum(1 for x in on if x) < 40
    for body, flag in zip(bodies, on, strict=True):
        assert (body["thinking"] != {"type": "disabled"}) == flag  # disabled exactly when the call was drawn off
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"} or not flag


async def test_a_share_of_one_never_touches_the_request_and_records_nothing() -> None:
    bodies, model = _sampled(1.0)
    await model.complete("s", "p", Answer)
    assert bodies[0]["thinking"] == {"type": "adaptive", "display": "summarized"} and model.calls[0].thinking_on is None


async def test_a_share_of_zero_disables_thinking_on_every_call() -> None:
    bodies, model = _sampled(0.0)
    for _ in range(5):
        await model.complete("s", "p", Answer)
    assert all(b["thinking"] == {"type": "disabled"} for b in bodies) and not any(c.thinking_on for c in model.calls)


def test_the_run_log_counts_thinking_calls_and_provenance_records_the_share() -> None:
    from harness.usage import api_call
    from harness.workflow.bench import provenance
    from harness.workflow.plan import Plan
    from harness.workflow.records import Recorder

    calls = [api_call("m", 1, 1, 0, 0, None, cost_usd=0.0, thinking_on=t) for t in (True, False)]
    assert Recorder().add_usage(calls)["thinking_on"] == 1
    assert "thinking_on" not in Recorder().add_usage([api_call("m", 1, 1, 0, 0, None, cost_usd=0.0)])
    plan = Plan(goal="g", behaviors=["b"])
    kw: dict[str, Any] = {"plan_path": None, "backend": "openrouter", "effort": None, "max_attempts": 3, "execution": "x", "bench": None}
    assert provenance(plan, thinking_sample=0.2, **kw)["thinking_sample"] == 0.2
    assert "thinking_sample" not in provenance(plan, **kw)


def test_an_anthropic_bearer_token_in_the_environment_never_reaches_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "leak-me")
    headers = make_openrouter_client(OpenRouterConfig()).auth_headers
    assert "Authorization" not in headers and "leak-me" not in json.dumps(dict(headers))


async def test_a_billed_reply_without_cost_is_recorded_before_the_run_stops() -> None:
    model, _ = make([message(cost=None)])
    with pytest.raises(ApiFailure):
        await model.complete("s", "p", Answer)
    assert len(model.calls) == 1 and model.calls[0].output_tokens == 5


async def test_an_openrouter_call_is_capped_at_the_configured_max_tokens() -> None:
    model, bodies = make([message()])
    await model.complete("s", "p", Answer)
    assert bodies[0]["max_tokens"] == 4096 == HarnessConfig().openrouter.max_tokens


# --- finish reason, wall time, slow-call warning ----------------------------------------------------------------------------

async def test_each_call_records_why_it_ended_and_how_long_it_took() -> None:
    model, _ = make([message()])
    await model.complete("s", "p", Answer)
    [call] = model.calls
    assert call.stop_reason == "end_turn" and call.seconds is not None and call.seconds >= 0


async def test_the_node_log_line_carries_the_stop_reasons_and_a_cap_hit_is_flagged() -> None:
    from harness.usage import api_call
    from harness.workflow.progress import describe
    from harness.workflow.records import Recorder

    calls = [api_call("m", 1, 1, 0, 0, None, cost_usd=0.0, stop_reason=r, seconds=s) for r, s in (("max_tokens", 100.0), ("end_turn", 5.0))]
    fields = Recorder().add_usage(calls)
    assert fields["stop_reason"] == "end_turn" and fields["stop_reasons"] == ["max_tokens", "end_turn"] and fields["call_seconds"] == 105.0
    assert "cut at the token cap x1" in describe({"node": "write_impl", "written": ["a.py"], **fields})


async def test_a_slow_request_warns_once_and_is_still_awaited() -> None:
    import asyncio

    warned: list[tuple[str, float]] = []

    model, _ = make([message()])
    real = model._create

    async def slow(params: dict[str, Any]) -> Any:
        await asyncio.sleep(0.15)
        return await real(params)

    model.slow_after_s, model.on_slow = 0.05, lambda m, s: warned.append((m, s))
    model._create = slow  # type: ignore[method-assign]
    assert (await model.complete("s", "p", Answer)).value == 3
    assert len(warned) == 1 and warned[0][0] == "z-ai/glm-5.3" and warned[0][1] >= 0.05
