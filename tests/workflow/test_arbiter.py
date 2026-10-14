"""F10 gates: the Arbiter, a stronger model that rules on a Dispute no Invariant upholds (ADR 0008).

No live calls: the Arbiter is the real ``AnthropicWorkflowModel`` (as ``backend.make_arbiter`` builds it) on an
``httpx2.MockTransport``, and the tester and implementer are scripted."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, TypeVar

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from harness.config import HarnessConfig, WorkflowConfig, arbiter_role
from harness.inspector import describe_node
from harness.sandbox import LocalSandbox, Workspace
from harness.usage import PRICING, api_call
from harness.workflow import cli, prompts
from harness.workflow.backend import make_arbiter
from harness.workflow.batch import Batcher, BatchedWorkflowModel, batch_params
from harness.workflow.cli import batched_arbiter, show
from harness.workflow.execute import EXECUTION, Ruling, build_execution_graph, execution_variant
from harness.workflow.invariants import Invariant
from harness.workflow.model import FALLBACK_BETA, AnthropicWorkflowModel, WorkflowModelError
from harness.workflow.progress import describe, final_report
from harness.workflow.records import write_summary
from tests.workflow.test_anthropic_model import message
from tests.workflow.test_api_runs import CostlyModel
from tests.workflow.test_execute import (
    BAD_MUL_TEST, MOD, MUL, MUL_IMPL, MUL_TEST, NEG, NEG_TEST, Run, ScriptedModel, dispute, edit, nodes, run_loop,
)

T = TypeVar("T", bound=BaseModel)

SONNET, OPUS, HAIKU = "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"
SAMPLING = {"temperature", "top_p", "top_k"}
MUL_RULE = Invariant(text="mul(a, b) is a times b", fn="mul", check="result == args[0] * args[1]")
WRONG = Ruling(verdict="test_wrong", reason="The goal says mul multiplies, and 2 times 3 is 6.", proposed_invariant=MUL_RULE)
RIGHT = Ruling(verdict="test_right", reason="The goal says mul multiplies; 6 is right, the implementer is wrong.")
UNCLEAR = Ruling(verdict="unclear", reason="The goal does not say what mul does with these values.", proposed_invariant=MUL_RULE)
THOUGHT = "The test asserts 7 for mul(2, 3); multiplying gives 6."


class Arbiter:
    """The Arbiter on a fake Messages API: each request gets the next Ruling, served by ``served`` (a fallback) or by
    the model asked for, with a thinking summary and a request id."""

    def __init__(self, rulings: list[Ruling], *, model: str = SONNET, served: str | None = None,
                 usage: dict[str, int] | None = None) -> None:
        self.rulings = rulings
        self.requests: list[httpx2.Request] = []
        self.bodies: list[dict[str, Any]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            self.bodies.append(json.loads(request.content))
            body = message(self.rulings[len(self.bodies) - 1].model_dump_json(), usage=usage)
            body["model"] = served or model
            body["content"].insert(0, {"type": "thinking", "thinking": THOUGHT, "signature": "sig"})
            return httpx2.Response(200, json=body, headers={"request-id": f"req_arb_{len(self.bodies)}"})

        client = anthropic.AsyncAnthropic(
            api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0,
        )
        model_ = make_arbiter(HarnessConfig(workflow=WorkflowConfig(arbiter=model)), client)
        assert isinstance(model_, AnthropicWorkflowModel)
        self.model = model_

    def prompts(self) -> list[str]:
        return [b["messages"][0]["content"] for b in self.bodies]


class Invalid:
    """An Arbiter whose reply is invalid twice over."""

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        raise WorkflowModelError("invalid JSON twice")


def routes(run: Run, source: str) -> list[dict[str, Any]]:
    return [r for r in run.log.lines() if r["node"] == "route" and r["from"] == source]


# --- gate 1: test_wrong goes to the write_test of the Behavior that wrote the test --------------------------------------

async def test_a_test_wrong_ruling_sends_the_test_back_to_the_behavior_that_wrote_it_and_the_corrected_test_reaches_green(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    wrong_mul = MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a * b + 1\n"  # passes the bad Behavior 1 test
    model = ScriptedModel([
        BAD_MUL_TEST, edit("src/mod.py", wrong_mul),  # Behavior 1 goes Green against a wrong test
        NEG_TEST, dispute("test_b1", "mul(2, 3) is 6", edit("src/mod.py", NEG).edits[0]),  # Behavior 2 disputes it
        MUL_TEST,  # Behavior 1's test, corrected
        edit("src/mod.py", NEG),  # Polish
    ])
    arbiter = Arbiter([WRONG])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], assumptions=[], arbiter=arbiter.model,
                         settings=WorkflowConfig(arbiter=SONNET))
    trace = run.rec.trace
    i = trace.index("check_dispute")
    assert trace[i:i + 7] == [
        "check_dispute", "arbitrate", "write_test", "run_red", "check_invariants", "resume_behavior", "run_green",
    ]
    rewrite = nodes(run, "write_test")[-1]
    assert rewrite["behavior"] == 0  # the Behavior that wrote the test, not the one that disputed it
    assert WRONG.reason in rewrite["prompt"] and "mul(2, 3) is 6" in rewrite["prompt"]
    assert [b.green for b in run.rec.behaviors] == [True, True]
    disputing = run.rec.behaviors[1]
    assert (disputing.arbitrations, disputing.rulings_test_wrong, disputing.rulings_test_right) == (1, 1, 0)
    assert disputing.attempts == 0 and disputing.disputes_upheld == 0  # not counted, and no Invariant upheld it
    assert "== 6" in (run.ws.path / "tests" / "test_behavior_1.py").read_text()
    # The proposed Invariant is shown at Sign-off and recorded, never added to the Plan.
    [proposed] = run.rec.proposed_invariants
    assert (proposed["text"], proposed["check"], proposed["test"], proposed["model"]) == (MUL_RULE.text, MUL_RULE.check, "test_b1", SONNET)
    sign_off = next(p for p in run.payloads if p["kind"] == "sign_off")
    assert sign_off["proposed_invariants"] == [proposed]
    shown = io.StringIO()
    show(sign_off, shown)
    assert f"proposed Invariant (from the Arbiter, on test_b1; not applied): {MUL_RULE.text}" in shown.getvalue()
    assert run.final["plan"]["invariants"] == []
    summary = json.loads(write_summary(
        tmp_path / "summaries", model="fake", project=fixture_repo, no_hints=False, status=run.final.get("status"),
        planned=2, rec=run.rec, workspace=tmp_path, run_log=run.log.path,
    ).read_text())
    assert summary["proposed_invariants"] == [proposed]
    t = summary["totals"]
    assert (t["arbitrations"], t["rulings_test_wrong"], t["rulings_test_right"]) == (1, 1, 0)
    assert summary["routes"]["model"] == 1


# --- gate 2: test_right puts the Ruling in the next write_impl prompt ---------------------------------------------------

async def test_a_test_right_ruling_reaches_the_next_write_impl_prompt(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "I believe mul(2, 3) is 5"), MUL_IMPL, edit("src/mod.py", MUL)])
    arbiter = Arbiter([RIGHT])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    assert run.rec.trace[2:8] == ["run_red", "write_impl", "check_dispute", "arbitrate", "run_green", "write_impl"]
    second = [p for p in model.prompts if "Task: make the tests pass" in p][1]
    assert "an Arbiter ruled the test right" in second and RIGHT.reason in second
    stats = run.rec.behaviors[0]
    assert stats.green and stats.attempts == 2 and (stats.arbitrations, stats.rulings_test_right) == (1, 1)
    [ruled] = nodes(run, "arbitrate")
    assert ruled["reply"]["verdict"] == "test_right" and ruled["arbiter"] == SONNET and ruled["served_by"] == SONNET


# --- gate 3: the bound ----------------------------------------------------------------------------------------------

async def test_the_bound_holds_a_second_dispute_in_the_same_behavior_takes_the_code_route(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([
        MUL_TEST, dispute("test_b1", "it should be 5"), dispute("test_b1", "it really should be 5"), MUL_IMPL,
        edit("src/mod.py", MUL),
    ])
    arbiter = Arbiter([RIGHT])  # a second request would find no Ruling
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    assert len(arbiter.bodies) == 1 and run.rec.trace.count("arbitrate") == 1
    first, second = routes(run, "check_dispute")
    assert (first["to"], first["field"], first["value"], first["basis"]) == ("arbitrate", "arbitrations", 0, "code")
    assert (second["to"], second["field"], second["value"], second["basis"]) == ("run_green", "arbitrations", 1, "code")
    assert run.rec.behaviors[0].green and run.rec.behaviors[0].disputes == 2


async def test_the_bound_is_per_behavior_and_configurable(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = ScriptedModel([
        MUL_TEST, dispute("test_b1", "5?"), dispute("test_b1", "5!"), MUL_IMPL,
        NEG_TEST, dispute("test_b2", "neg(3) is 3"), edit("src/mod.py", NEG), edit("src/mod.py", NEG),
    ])
    arbiter = Arbiter([RIGHT, RIGHT, RIGHT])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], assumptions=[], arbiter=arbiter.model,
                         settings=WorkflowConfig(max_arbitrations=2, max_attempts=4))
    assert [b.arbitrations for b in run.rec.behaviors] == [2, 1]  # the count starts again for Behavior 2
    assert len(arbiter.bodies) == 3


# --- gate 4: a model-basis Route record on every arbitration ------------------------------------------------------------

async def test_every_arbitration_writes_a_model_basis_route_record_with_its_request_id_reason_and_thinking(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "it should be 5"), MUL_IMPL, edit("src/mod.py", MUL)])
    arbiter = Arbiter([RIGHT])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    [route] = routes(run, "arbitrate")
    assert {k: route[k] for k in ("to", "basis", "field", "value", "model", "request_id", "reason", "thinking")} == {
        "to": "run_green", "basis": "model", "field": "verdict", "value": "test_right", "model": SONNET,
        "request_id": "req_arb_1", "reason": RIGHT.reason, "thinking": THOUGHT,
    }
    assert run.rec.routes["model"] == 1
    [ruled] = nodes(run, "arbitrate")
    assert ruled["request_id"] == "req_arb_1" and ruled["thinking"] == THOUGHT


async def test_an_unclear_ruling_goes_to_stuck_and_shows_the_ruling(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "it should be 5")])
    arbiter = Arbiter([UNCLEAR])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["abort"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    [stuck] = run.payloads
    assert stuck["kind"] == "stuck" and stuck["dispute"]["ruling"]["verdict"] == "unclear"
    [route] = routes(run, "arbitrate")
    assert (route["to"], route["basis"], route["value"]) == ("stuck", "model", "unclear")
    shown = io.StringIO()
    show(stuck, shown)
    text = shown.getvalue()
    assert f"the Arbiter ({SONNET}) ruled unclear: {UNCLEAR.reason}" in text
    assert "not applied): mul(a, b) is a times b" in text


async def test_an_arbiter_with_no_valid_ruling_leaves_the_dispute_not_upheld(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "it should be 5"), MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=Invalid())
    [route] = routes(run, "arbitrate")
    assert (route["to"], route["basis"], route["field"]) == ("run_green", "code", "ruling")
    assert "not upheld" in [p for p in model.prompts if "Task: make the tests pass" in p][1]
    assert run.rec.behaviors[0].green


# --- gate 5: no implementation file in the Arbiter's prompt -------------------------------------------------------------

async def test_the_arbiters_prompt_has_the_goal_rules_and_test_but_no_implementation_file(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    canary = "# braid-impl-canary-7f3a"
    implementation = MUL.replace("return a * b", f"return a * b  {canary}")
    model = ScriptedModel([
        MUL_TEST, dispute("test_b1", "it should be 5", edit("src/mod.py", implementation).edits[0]),
        MUL_IMPL, edit("src/mod.py", MUL),
    ])
    arbiter = Arbiter([RIGHT])
    rule = Invariant(text="mul is never negative for naturals", fn="mul", check="result >= 0")
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model, invariants=[rule])
    assert run.rec.behaviors[0].arbitrations == 1
    [body] = arbiter.bodies
    sent = json.dumps(body)
    assert canary not in sent and "return a + b" not in sent and "def add" not in sent  # no implementation file
    [prompt] = arbiter.prompts()
    assert "Goal: arithmetic helpers" in prompt and rule.text in prompt and "assert mul(2, 3) == 6" in prompt
    assert "tests/test_behavior_1.py" in prompt and "The implementer's reason: it should be 5" in prompt
    assert body["system"][0]["text"] == prompts.ARBITER_SYSTEM


# --- gate 6: no arbiter, no change ------------------------------------------------------------------------------------

async def test_with_no_arbiter_the_graph_and_the_variant_are_unchanged(fixture_repo: Path, tmp_path: Path) -> None:
    ws = await Workspace.create(fixture_repo, tmp_path / "workspaces", "run1")
    plain = build_execution_graph(ScriptedModel([]), LocalSandbox(), ws, WorkflowConfig())
    judged = build_execution_graph(ScriptedModel([]), LocalSandbox(), ws, WorkflowConfig(arbiter=SONNET), arbiter=ScriptedModel([]))
    assert "arbitrate" not in plain.nodes and set(judged.nodes) == set(plain.nodes) | {"arbitrate"}
    assert plain.branches["check_dispute"].keys() == judged.branches["check_dispute"].keys()
    assert make_arbiter(HarnessConfig()) is None and arbiter_role(WorkflowConfig()) is None
    assert execution_variant(WorkflowConfig()) == EXECUTION
    assert execution_variant(WorkflowConfig(arbiter=OPUS)) == f"{EXECUTION}/arbiter={OPUS}"


async def test_with_no_arbiter_an_uncovered_dispute_goes_straight_to_run_green(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "wrong"), MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"], assumptions=[])
    [route] = routes(run, "check_dispute")
    assert (route["to"], route["field"], route["value"]) == ("run_green", "phase", "impl")  # the F3 route record
    assert "arbitrate" not in run.rec.trace and run.rec.roles == {}


# --- pricing, the Spend cap and cost per Role -------------------------------------------------------------------------

def test_sonnet_and_opus_are_priced_with_an_explicit_cache_read_price_and_haiku_is_unchanged() -> None:
    m = 1_000_000
    assert api_call(OPUS, m, 0, 0, 0, None).cost_usd == pytest.approx(4.00)
    assert api_call(OPUS, 0, m, 0, 0, None).cost_usd == pytest.approx(20.00)
    assert api_call(OPUS, 0, 0, m, 0, None).cost_usd == pytest.approx(0.20)  # 0.05x input, not 0.1x
    assert api_call(OPUS, 0, 0, 0, m, None).cost_usd == pytest.approx(5.00)  # writes: 1.25x input
    assert api_call(SONNET, m, m, m, 0, None).cost_usd == pytest.approx(2.00 + 10.00 + 0.20)
    assert api_call(OPUS, 150_000, 0, 0, 0, None).cost_usd == pytest.approx(0.60)  # no long tier
    assert api_call(HAIKU, 0, 0, 50_000, 0, None).cost_usd == pytest.approx(50_000 * 0.10 * 0.1 / m)
    assert api_call(HAIKU, 0, 0, 150_000, 0, None).cost_usd == pytest.approx(150_000 * 0.50 * 0.1 / m)  # long tier
    assert PRICING[HAIKU].cache_read is None


async def test_the_arbiters_cost_counts_toward_the_spend_cap_and_is_reported_per_role(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    usage = {"input_tokens": 30_000, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    arbiter = Arbiter([RIGHT], model=OPUS, usage=usage)  # $0.12, over the default $0.10 cap
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "it should be 5")])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, [], behaviors=["mul multiplies"], assumptions=[],
                         arbiter=arbiter.model)
    assert run.final["status"] == "spend_cap"
    [route] = routes(run, "arbitrate")
    assert (route["to"], route["field"]) == ("spend_cap", "cost_usd")
    assert run.rec.roles == {"arbiter": {"calls": 1, "cost_usd": pytest.approx(0.12)}}
    summary = json.loads(write_summary(
        tmp_path / "summaries", model="fake", project=fixture_repo, no_hints=False, status="spend_cap", planned=1,
        rec=run.rec, workspace=tmp_path, run_log=run.log.path,
    ).read_text())
    assert summary["roles"] == {"arbiter": {"calls": 1, "cost_usd": pytest.approx(0.12)}}
    assert summary["totals"]["cost_usd"] == pytest.approx(0.12)


async def test_api_calls_are_reported_per_role(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    model = CostlyModel([MUL_TEST, dispute("test_b1", "it should be 5"), MUL_IMPL, edit("src/mod.py", MUL)], price=0.001)
    arbiter = Arbiter([RIGHT])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    roles = run.rec.roles
    assert roles["tester"]["calls"] == 1 and roles["implementer"]["calls"] == 3 and roles["arbiter"]["calls"] == 1
    assert sum(r["cost_usd"] for r in roles.values()) == pytest.approx(run.rec.cost_usd)
    report = final_report(run.rec, 1.0)
    assert "by role: arbiter $" in report and "implementer $0.0030 (3 calls)" in report
    assert f"ruled test_right ({SONNET})" in describe(nodes(run, "arbitrate")[0])


# --- refusal fallbacks ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arbiter_model", [SONNET, OPUS])
async def test_sonnet_and_opus_arbiters_opt_in_to_server_side_fallbacks(arbiter_model: str) -> None:
    arbiter = Arbiter([RIGHT], model=arbiter_model)
    assert await arbiter.model.complete("sys", "rule", Ruling) == RIGHT
    [request], [body] = arbiter.requests, arbiter.bodies
    assert request.url.path == "/v1/messages" and request.url.params.get("beta") == "true"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["fallbacks"] == "default" and body["model"] == arbiter_model
    assert not SAMPLING & body.keys() and "tool_choice" not in body


async def test_a_fallback_records_the_model_that_served_the_request_and_prices_it_as_that_model(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    arbiter = Arbiter([RIGHT], model=OPUS, served=SONNET)
    model = ScriptedModel([MUL_TEST, dispute("test_b1", "it should be 5"), MUL_IMPL, edit("src/mod.py", MUL)])
    run = await run_loop(fixture_repo, fixture_venv, tmp_path, model, ["yes", "no"], behaviors=["mul multiplies"],
                         assumptions=[], arbiter=arbiter.model)
    [call] = arbiter.model.calls
    assert call.served_model == SONNET and call.cost_usd == pytest.approx((10 * 2.00 + 5 * 10.00) / 1_000_000)
    [ruled] = nodes(run, "arbitrate")
    assert (ruled["arbiter"], ruled["served_by"]) == (OPUS, SONNET)
    assert routes(run, "arbitrate")[0]["model"] == SONNET  # the model that made the claim


async def test_only_the_arbiters_sonnet_or_opus_requests_carry_fallbacks() -> None:
    haiku = Arbiter([RIGHT], model=HAIKU)
    await haiku.model.complete("sys", "rule", Ruling)
    assert "fallbacks" not in haiku.bodies[0] and "beta" not in haiku.requests[0].url.params
    assert "anthropic-beta" not in haiku.requests[0].headers
    assert haiku.model.calls[0].served_model == HAIKU


def test_a_batched_arbiter_rides_the_runs_lane_and_never_sends_fallbacks() -> None:
    batcher = Batcher(anthropic.AsyncAnthropic(api_key="test-key"))
    model = BatchedWorkflowModel(batcher.lane(), HAIKU)
    config = HarnessConfig(workflow=WorkflowConfig(arbiter=OPUS, arbiter_effort="medium"))
    arbiter = batched_arbiter(model, config)
    assert isinstance(arbiter, BatchedWorkflowModel) and arbiter.lane is model.lane
    assert arbiter.model_id == OPUS and arbiter.effort == "medium" and not arbiter.fallbacks
    assert batched_arbiter(model, HarnessConfig()) is None
    assert batch_params({"model": OPUS, "fallbacks": "default", "betas": [FALLBACK_BETA]}) == {"model": OPUS}


def test_an_ollama_or_unpriced_arbiter_is_refused_before_a_run() -> None:
    with pytest.raises(ValueError, match="Anthropic .*or OpenRouter"):
        make_arbiter(HarnessConfig(workflow=WorkflowConfig(arbiter="gemma4:e4b")))
    with pytest.raises(ValueError, match="no pricing"):
        make_arbiter(HarnessConfig(workflow=WorkflowConfig(arbiter="claude-unpriced-9")), anthropic.AsyncAnthropic(api_key="k"))
    args = cli.build_parser().parse_args(["bench", "claude-plans", "--model", HAIKU])
    with pytest.raises(ValueError, match="no pricing"):
        cli.grid_cells(args, HarnessConfig(workflow=WorkflowConfig(arbiter="claude-unpriced-9")), lambda _c, _m: lambda: ScriptedModel([]))


def test_the_run_inspector_shows_an_arbitrate_line() -> None:
    line = {"node": "arbitrate", "behavior": 0, "prompt": "p", "reply": RIGHT.model_dump(), "thinking": THOUGHT}
    assert RIGHT.reason in describe_node(line) and THOUGHT in describe_node(line)
