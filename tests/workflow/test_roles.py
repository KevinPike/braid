"""F11 gates: Roles, each with its own model, effort, system prompt and View (adversarial TDD).

No live calls: Role models are the real ``AnthropicWorkflowModel`` (as ``backend.make_roles`` builds them) on one
``httpx2.MockTransport`` that answers each model from its own script, or ``BatchedWorkflowModel``s on a fake batch
endpoint, or scripted models."""

from __future__ import annotations

import io
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, ValidationError

from harness.config import HarnessConfig, RoleConfig, RolesConfig, WorkflowConfig, arbiter_role, role_model, role_view
from harness.workflow import cli, prompts, views
from harness.workflow.backend import make_roles
from harness.workflow.batch import BATCH_DISCOUNT, BatchedWorkflowModel, Batcher
from harness.workflow.cli import batched_roles
from harness.workflow.execute import EXECUTION, Edits, FileEdit, execution_variant
from harness.workflow.model import AnthropicWorkflowModel
from harness.workflow.progress import final_report
from harness.workflow.plan import Plan
from harness.workflow.records import Recorder
from harness.sandbox import LocalSandbox
from harness.usage import api_call
from tests.anthropic_fakes import FakeBatches, succeeded
from tests.conftest import git
from tests.workflow.roles_baseline import BASELINE, RIGHT, scripted_prompts
from tests.workflow.test_anthropic_model import message
from tests.workflow.test_api_runs import cfg
from tests.workflow.test_execute import MUL, MUL_IMPL, MUL_TEST, NEG_TEST, Run, ScriptedModel, dispute, edit, run_loop

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
CANARY = "CANARY-BODY-7f3a"
CANARY_MOD = (
    '"""Arithmetic helpers."""\n\n\ndef add(a: int, b: int) -> int:\n    """Adds two numbers."""\n'
    f'    canary = "{CANARY}"\n    return a + b if canary else 0\n\n\n'
    'def boom() -> int:\n    """Raises."""\n'
    f'    secret = "{CANARY}-2"\n    raise RuntimeError("boom " + str(len(secret)))\n'
)
CANARY_MUL = CANARY_MOD + "\n\ndef mul(a: int, b: int) -> int:\n    return a * b\n"
CANARY_NEG = CANARY_MUL + "\n\ndef neg(a: int) -> int:\n    return -a\n"
RATIONALE = "TESTER-RATIONALE: mul(2, 3) is the smallest case a wrong mul gets wrong"
INTERFACE = WorkflowConfig(roles=RolesConfig(tester=RoleConfig(view="interface")))


def canary_repo(fixture_repo: Path) -> Path:
    (fixture_repo / "src" / "mod.py").write_text(CANARY_MOD)
    git(fixture_repo, "commit", "-qam", "canary")
    return fixture_repo


def rationale_edit(path: str, content: str) -> Edits:
    return Edits(edits=[FileEdit(path=path, content=content)], rationale=RATIONALE)


async def canary_run(fixture_repo: Path, fixture_venv: Path, tmp_path: Path, settings: WorkflowConfig) -> tuple[Run, list[str], list[str]]:
    """A test that blows up inside the implementation (its traceback quotes the canary body), then two Behaviors."""
    exploding = rationale_edit("tests/test_behavior_1.py", "from src.mod import boom\n\nboom()\n")
    mul_test = Edits(edits=MUL_TEST.edits, rationale=RATIONALE)
    model = ScriptedModel([
        exploding, mul_test, edit("src/mod.py", CANARY_MUL), Edits(edits=NEG_TEST.edits, rationale=RATIONALE),
        edit("src/mod.py", CANARY_NEG), edit("src/mod.py", CANARY_NEG),
    ])
    run = await run_loop(canary_repo(fixture_repo), fixture_venv, tmp_path, model, ["yes", "no"], assumptions=[],
                         settings=settings)
    tester = [r["prompt"] for r in run.log.lines() if r["node"] == "write_test"]
    implementer = [r["prompt"] for r in run.log.lines() if r["node"] in ("write_impl", "polish")]
    return run, tester, implementer


# --- gate 1: the interface View shows the tester no function body --------------------------------------------------

async def test_with_the_interface_view_no_tester_prompt_contains_a_function_body(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    run, tester, implementer = await canary_run(fixture_repo, fixture_venv, tmp_path, INTERFACE)
    assert [b.green for b in run.rec.behaviors] == [True, True]
    assert len(tester) == 3 and not any(CANARY in p for p in tester)
    assert all(prompts.INTERFACE_NOTE in p and "def add(a: int, b: int) -> int:" in p and "Adds two numbers." in p for p in tester)
    assert "def mul(a: int, b: int) -> int:\n    ..." in tester[2]  # the new code, as its signature
    assert "from src.mod import mul" in tester[2]  # earlier tests are shown whole
    assert views.HIDDEN in tester[1] and "RuntimeError" in tester[1]  # the rejected test's traceback, redacted
    assert all(CANARY in p and prompts.INTERFACE_NOTE not in p for p in implementer)  # the implementer sees it all


async def test_with_the_full_view_the_tester_sees_the_bodies_the_interface_view_hides(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    """The control for gate 1: the same run in the ``full`` View, so the canary was there to leak."""
    _, tester, _ = await canary_run(fixture_repo, fixture_venv, tmp_path, WorkflowConfig(roles=RolesConfig(tester=RoleConfig(view="full"))))
    assert all(CANARY in p for p in tester)
    assert f'secret = "{CANARY}-2"' in tester[1]  # pytest's traceback quotes the body


async def test_the_implementer_never_sees_the_testers_rationale(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    run, _, implementer = await canary_run(fixture_repo, fixture_venv, tmp_path, INTERFACE)
    assert implementer and not any(RATIONALE in p for p in implementer)
    assert any(r["node"] == "write_test" and r["reply"]["rationale"] == RATIONALE for r in run.log.lines())  # only logged


def test_the_interface_keeps_signatures_docstrings_and_hints_and_drops_bodies_and_private_names() -> None:
    source = '''"""Module doc."""
from dataclasses import dataclass
LIMIT: int = 10
TABLE = {"a": 1}
_cache = {}


@dataclass
class Point:
    """A point."""
    x: int = 0

    def __init__(self, x: int) -> None:
        self.x = x * 2

    def norm(self) -> float:
        """Its length."""
        return abs(self.x)

    def _helper(self) -> None:
        print("hidden")


async def fetch(url: str, *, retries: int = 3) -> bytes:
    return b"secret body"


def _private() -> None:
    pass
'''
    shown = views.interface(source)
    assert '"""Module doc."""' in shown and "from dataclasses import dataclass" in shown
    assert "LIMIT: int = ..." in shown and "TABLE = ..." in shown and "_cache" not in shown
    assert "@dataclass\nclass Point:\n    \"\"\"A point.\"\"\"\n    x: int = ..." in shown
    assert "def __init__(self, x: int) -> None:\n        ..." in shown
    assert 'def norm(self) -> float:\n        """Its length."""\n        ...' in shown
    assert "async def fetch(url: str, *, retries: int=3) -> bytes:\n    ..." in shown
    for hidden in ("x * 2", "abs(", "_helper", "hidden", "secret body", "_private"):
        assert hidden not in shown


def test_a_file_that_does_not_parse_shows_nothing_of_its_contents() -> None:
    broken = "def f(:\n    return 'CANARY'\n"
    assert views.shown("src/mod.py", broken) == "# This file does not parse (line 1), so its interface cannot be shown.\n"
    assert views.shown("tests/test_x.py", broken) == broken  # a test file is shown whole
    assert views.shown("README.md", "# hi\n") == "# hi\n"
    assert "return 'CANARY'" in views.body_lines({"src/mod.py": broken})  # so its lines are redacted too
    out = "E   x\n>       return 'CANARY'\nE   ValueError: no\n"
    assert views.redact(out, views.body_lines({"src/mod.py": broken})) == f"E   x\n    {views.HIDDEN}\nE   ValueError: no\n"


# --- gate 2: each Role's requests go to its own model --------------------------------------------------------------

class Transport:
    """The Messages API on one ``httpx2.MockTransport``: each model answers from its own script and is counted."""

    def __init__(self, scripts: dict[str, list[BaseModel]]) -> None:
        self.scripts = scripts
        self.bodies: list[dict[str, Any]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            self.bodies.append(body)
            reply = message(self.scripts[body["model"]].pop(0).model_dump_json())
            reply["model"] = body["model"]
            return httpx2.Response(200, json=reply, headers={"request-id": f"req_{len(self.bodies)}"})

        self.client = anthropic.AsyncAnthropic(
            api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0,
        )

    def per_model(self) -> Counter[str]:
        return Counter(b["model"] for b in self.bodies)

    def systems(self, model: str) -> set[str]:
        return {b["system"][0]["text"] for b in self.bodies if b["model"] == model}


ROLES = WorkflowConfig(roles=RolesConfig(
    tester=RoleConfig(model=SONNET, effort="low", view="interface"), arbiter=RoleConfig(model=OPUS),
))


async def test_each_roles_requests_go_to_its_own_model(fixture_repo: Path, fixture_venv: Path, tmp_path: Path) -> None:
    fake = Transport({
        SONNET: [MUL_TEST],
        HAIKU: [dispute("test_b1", "mul(2, 3) is 5"), edit("src/mod.py", CANARY_MUL), edit("src/mod.py", CANARY_MUL)],
        OPUS: [RIGHT],
    })
    config = HarnessConfig(workflow=ROLES)
    run_model = AnthropicWorkflowModel(fake.client, HAIKU)
    roles = make_roles(config, HAIKU, fake.client)
    assert set(roles) == {"tester", "arbiter"}  # the implementer runs on the run's model
    run = await run_loop(canary_repo(fixture_repo), fixture_venv, tmp_path, run_model, ["yes", "no"],
                         behaviors=["mul multiplies"], assumptions=[], settings=ROLES, arbiter=roles["arbiter"],
                         tester=roles["tester"])
    assert run.rec.behaviors[0].green
    assert fake.per_model() == {SONNET: 1, HAIKU: 3, OPUS: 1}
    assert fake.systems(SONNET) == {prompts.TESTER_SYSTEM}  # each Role its own cache prefix
    assert fake.systems(HAIKU) == {prompts.SYSTEM} and fake.systems(OPUS) == {prompts.ARBITER_SYSTEM}
    assert [b["output_config"].get("effort") for b in fake.bodies if b["model"] == SONNET] == ["low"]
    assert not any(CANARY in b["messages"][0]["content"] for b in fake.bodies if b["model"] == SONNET)
    assert run.rec.roles == {
        "tester": {"calls": 1, "cost_usd": pytest.approx(api_call(SONNET, 10, 5, 0, 0, None).cost_usd)},
        "implementer": {"calls": 3, "cost_usd": pytest.approx(3 * api_call(HAIKU, 10, 5, 0, 0, None).cost_usd)},
        "arbiter": {"calls": 1, "cost_usd": pytest.approx(api_call(OPUS, 10, 5, 0, 0, None).cost_usd)},
    }


async def test_batched_roles_share_the_runs_lane_and_each_request_names_its_roles_model(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    """F12's lane is per run: a batch may mix models, and a run's calls are sequential whichever Role makes them."""
    scripts: dict[str, list[BaseModel]] = {SONNET: [MUL_TEST], HAIKU: [edit("src/mod.py", CANARY_MUL)] * 2}

    def answer(_cid: str, params: dict[str, Any]) -> dict[str, Any]:
        result = succeeded(scripts[params["model"]].pop(0).model_dump_json())
        result["message"]["model"] = params["model"]
        return result

    fake = FakeBatches(answer, polls=0)
    batcher = Batcher(fake.client(), flush_s=60, poll_s=0)
    model = BatchedWorkflowModel(batcher.lane(), HAIKU)
    config = HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(tester=RoleConfig(model=SONNET))))
    roles = batched_roles(model, config)
    assert roles is not None and set(roles) == {"tester"}
    tester = roles["tester"]
    assert isinstance(tester, BatchedWorkflowModel) and tester.lane is model.lane and tester.model_id == SONNET
    run = await run_loop(canary_repo(fixture_repo), fixture_venv, tmp_path, model, ["yes", "no"],
                         behaviors=["mul multiplies"], assumptions=[], settings=config.workflow, tester=tester)
    model.close()
    sent = [r for batch in fake.batches for r in batch]
    assert [r["params"]["model"] for r in sent] == [SONNET, HAIKU, HAIKU]
    assert [r["custom_id"] for r in sent] == ["run1-1", "run1-2", "run1-3"]  # one lane, one count
    assert run.rec.roles["tester"]["cost_usd"] == pytest.approx(
        api_call(SONNET, 1000, 200, 0, 0, None).cost_usd * BATCH_DISCOUNT)
    assert batched_roles(AnthropicWorkflowModel(fake.client(), HAIKU), config) is None


def test_a_role_model_must_be_an_anthropic_model_with_a_price() -> None:
    with pytest.raises(ValueError, match=r"roles.tester\] needs an Anthropic .*or OpenRouter"):
        make_roles(HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(tester=RoleConfig(model="gemma4:e4b")))), HAIKU)
    with pytest.raises(ValueError, match=r"roles.implementer\] needs an Anthropic .*or OpenRouter"):  # an effort on Ollama
        make_roles(HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(implementer=RoleConfig(effort="low")))), "gemma4:e4b")
    with pytest.raises(ValueError, match="no pricing"):
        make_roles(HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(tester=RoleConfig(model="claude-unpriced-9")))),
                   HAIKU, anthropic.AsyncAnthropic(api_key="k"))
    args = cli.build_parser().parse_args(["bench", "exotic", "--model", HAIKU])
    unpriced = HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(implementer=RoleConfig(model="claude-unpriced-9"))))
    with pytest.raises(ValueError, match="no pricing"):
        cli.grid_cells(args, unpriced, lambda _c, _m: lambda: ScriptedModel([]))
    assert make_roles(HarnessConfig(workflow=WorkflowConfig(roles=RolesConfig(tester=RoleConfig(view="interface")))), HAIKU) == {}


# --- gate 3: cost and calls per Role in the summary, and the Roles in its provenance ---------------------------------

async def test_the_summary_carries_cost_and_calls_per_role_and_the_provenance_names_each_roles_model_and_view(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    fake = Transport({SONNET: [MUL_TEST], HAIKU: [MUL_IMPL, edit("src/mod.py", MUL)]})
    config = cfg(tmp_path, fixture_venv, effort="medium", roles=RolesConfig(tester=RoleConfig(model=SONNET, view="interface")))
    out = io.StringIO()
    result = await cli.build_once(
        Plan(goal="g", behaviors=["mul multiplies"]), fixture_repo, model=AnthropicWorkflowModel(fake.client, HAIKU, effort="medium"),
        model_id=HAIKU, sandbox=LocalSandbox(), config=config, hints=False, ask=lambda _p: "no", out=out, err=io.StringIO(),
        roles=make_roles(config, HAIKU, fake.client),
    )
    assert result.summary is not None and fake.per_model() == {SONNET: 1, HAIKU: 2}
    summary = json.loads(result.summary.read_text())
    tester, implementer = api_call(SONNET, 10, 5, 0, 0, None).cost_usd, api_call(HAIKU, 10, 5, 0, 0, None).cost_usd
    assert summary["roles"] == {
        "tester": {"calls": 1, "cost_usd": pytest.approx(tester)},
        "implementer": {"calls": 2, "cost_usd": pytest.approx(2 * implementer)},
    }
    assert summary["totals"]["cost_usd"] == pytest.approx(tester + 2 * implementer)
    assert summary["execution"] == f"{EXECUTION}/tester={SONNET}@medium:interface/implementer={HAIKU}@medium:full"
    text = out.getvalue()
    assert f"tester: {SONNET}, interface view\nimplementer: {HAIKU}, full view\n" in text
    assert "by role: implementer $" in text and "tester $" in text
    assert "by role" not in final_report(Recorder(roles={"tester": {"calls": 1, "cost_usd": 0.1}}, cost_usd=0.1), 1.0)


def test_the_execution_variant_names_the_roles_only_once_they_are_separated() -> None:
    assert execution_variant(WorkflowConfig(), HAIKU) == EXECUTION
    arbiter = WorkflowConfig(roles=RolesConfig(arbiter=RoleConfig(model=OPUS)))
    assert execution_variant(arbiter, HAIKU) == f"{EXECUTION}/arbiter={OPUS}"  # F10's suffix, from either key
    both = WorkflowConfig(arbiter=SONNET, roles=RolesConfig(implementer=RoleConfig(model=SONNET, effort="high")))
    assert execution_variant(both, HAIKU) == f"{EXECUTION}/tester={HAIKU}:full/implementer={SONNET}@high:full/arbiter={SONNET}"


# --- gate 4: with no Roles configured, every prompt is byte-identical to before F11 ---------------------------------

async def test_with_no_roles_configured_every_prompt_is_byte_identical_to_before(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    baseline = json.loads(BASELINE.read_text())  # captured at 15b02c0, before Roles (``roles_baseline.py``)
    assert await scripted_prompts(fixture_repo, fixture_venv, tmp_path) == baseline
    assert {p["system"] for p in baseline if p["role"] == "model"} == {prompts.SYSTEM}  # one cache prefix, as before


async def test_the_arbiter_set_under_roles_sends_the_same_prompts_as_f10s_key(
    fixture_repo: Path, fixture_venv: Path, tmp_path: Path,
) -> None:
    settings = WorkflowConfig(roles=RolesConfig(arbiter=RoleConfig(model=SONNET)))
    assert await scripted_prompts(fixture_repo, fixture_venv, tmp_path, settings) == json.loads(BASELINE.read_text())


# --- config ------------------------------------------------------------------------------------------------------------

def test_roles_load_from_toml_and_unset_fields_fall_back_to_workflow() -> None:
    config = HarnessConfig.model_validate(tomllib.loads('''
[workflow]
effort = "low"
arbiter = "claude-sonnet-5-5"
arbiter_effort = "high"

[workflow.roles.tester]
model = "claude-opus-5-5"
view = "interface"

[workflow.roles.arbiter]
model = "claude-opus-5-5"
'''))
    wf = config.workflow
    assert wf.roles.separated and not wf.roles.implementer.configured
    assert role_model(wf, "tester", HAIKU) == role_model(wf, "tester", SONNET)
    assert (role_model(wf, "tester", HAIKU).model, role_model(wf, "tester", HAIKU).effort) == (OPUS, "low")
    assert (role_model(wf, "implementer", HAIKU).model, role_model(wf, "implementer", HAIKU).effort) == (HAIKU, "low")
    assert (role_view(wf, "tester"), role_view(wf, "implementer"), role_view(wf, "arbiter")) == ("interface", "full", "full")
    judge = arbiter_role(wf)  # [workflow.roles.arbiter] wins field by field; [workflow] arbiter_effort fills the rest
    assert judge is not None and (judge.model, judge.effort) == (OPUS, "high")
    assert not WorkflowConfig().roles.separated and role_view(WorkflowConfig(), "tester") == "full"


def test_only_the_tester_takes_the_interface_view() -> None:
    with pytest.raises(ValidationError, match="implementer edits the implementation"):
        RolesConfig(implementer=RoleConfig(view="interface"))
    with pytest.raises(ValidationError, match="never sees the implementation"):
        RolesConfig(arbiter=RoleConfig(view="full"))
