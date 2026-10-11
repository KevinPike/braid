"""F12 gates: bench runs in lockstep through Message Batches at batch prices, and a model x effort sweep with its
frontier report. The batch endpoints are faked on ``httpx2.MockTransport``; nothing reaches the network."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from harness.anthropic_api import ApiFailure
from harness.config import HarnessConfig
from harness.sandbox import LocalSandbox
from harness.usage import api_call
from harness.workflow import bench
from harness.workflow.backend import Choice
from harness.workflow.batch import BATCH_DISCOUNT, BatchedWorkflowModel, Batcher, batch_params
from harness.workflow.cli import Cell, build_parser, grid_cells, report, run_grid
from harness.workflow.model import AnthropicWorkflowModel, WorkflowModel
from harness.workflow.records import read_log
from tests.anthropic_fakes import MODEL, FakeBatches, errored, expired, succeeded
from tests.workflow.test_anthropic_model import Answer, message
from tests.workflow.test_api_runs import cfg
from tests.workflow.test_bench import line, replies, two_katas
from tests.workflow.test_execute import ScriptedModel

USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def scripted(script: list[BaseModel], fail: dict[str, dict[str, Any]] | None = None) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Each run gets ``script`` in its own call order: ``run2-3`` is run 2's third call. ``fail`` overrides a call."""
    def answer(custom_id: str, _params: dict[str, Any]) -> dict[str, Any]:
        if fail and custom_id in fail:
            return fail[custom_id]
        n = int(custom_id.rsplit("-", 1)[1])
        return succeeded(script[n - 1].model_dump_json(), USAGE)

    return answer


def batched_cell(batcher: Batcher, config: HarnessConfig, label: str = "batched") -> Cell:
    def make() -> WorkflowModel:
        return BatchedWorkflowModel(batcher.lane(), MODEL)

    return Cell(label, Choice("anthropic", MODEL), config, make)


# --- gate 1: lockstep through one batch endpoint, results matched by custom_id in any order ---------------------------

async def test_runs_wait_for_each_other_and_each_gets_its_own_result() -> None:
    fake = FakeBatches(lambda cid, _p: succeeded(json.dumps({"value": int(cid[3]) * 10 + int(cid[-1]), "why": cid})))
    batcher = Batcher(fake.client(), flush_s=60, poll_s=0)
    got: dict[int, list[int]] = {}

    async def run(k: int) -> None:
        model = BatchedWorkflowModel(batcher.lane(), MODEL)
        got[k] = []
        for _ in range(3):
            for _ in range(k * 3):  # out of step between calls: no run starts a node at the same time as another
                await asyncio.sleep(0)
            got[k].append((await model.complete("sys", "p", Answer)).value)
        model.close()

    await asyncio.gather(*(run(k) for k in (1, 2, 3)))
    assert [len(b) for b in fake.batches] == [3, 3, 3]  # every call of a node in one batch, none left behind
    assert got == {1: [11, 12, 13], 2: [21, 22, 23], 3: [31, 32, 33]}  # the fake answers in reverse order
    assert fake.retrievals == {0: 2, 1: 2, 2: 2}  # one poll while in progress, then the one ``results`` makes


async def test_a_batch_goes_after_flush_s_when_a_run_is_not_waiting() -> None:
    fake = FakeBatches(lambda cid, _p: succeeded('{"value": 1, "why": "x"}'), polls=0)
    batcher = Batcher(fake.client(), flush_s=0.05, poll_s=0)
    busy = batcher.lane()  # an open run that is running its tests, not calling
    model = BatchedWorkflowModel(batcher.lane(), MODEL)
    assert (await asyncio.wait_for(model.complete("sys", "p", Answer), 5)).value == 1
    assert [len(b) for b in fake.batches] == [1]
    busy.close()


async def test_bench_runs_go_in_lockstep_through_batches(fixture_venv: Path, tmp_path: Path) -> None:
    fake = FakeBatches(scripted(replies()))
    batcher = Batcher(fake.client(), flush_s=60, poll_s=0)
    ledger = tmp_path / "bench.jsonl"
    out = io.StringIO()
    code = await run_grid(
        two_katas(tmp_path), [batched_cell(batcher, cfg(tmp_path, fixture_venv))], sandbox=LocalSandbox(), repeat=2,
        max_cost=1.0, ledger=ledger, parallel=4, out=out, err=io.StringIO(),
    )
    assert code == 0, out.getvalue()
    assert [len(b) for b in fake.batches] == [4, 4, 4]  # write_test, write_impl, polish: one batch per node, all 4 runs
    assert sorted(r["custom_id"] for r in fake.batches[1]) == ["run1-2", "run2-2", "run3-2", "run4-2"]
    lines = bench.read_ledger(ledger)
    assert len(lines) == 4 and all(x["summary"]["totals"]["green"] == 1 for x in lines)
    assert sorted((x["kata"], x["repeat"]) for x in lines) == [("alpha", 1), ("alpha", 2), ("beta", 1), ("beta", 2)]
    assert not any((tmp_path / "ws").iterdir()) and batcher.active == 0


async def test_a_batched_request_carries_what_a_single_call_sends() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=message('{"value": 1, "why": "x"}'))

    client = anthropic.AsyncAnthropic(api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    await AnthropicWorkflowModel(client, MODEL, effort="low").complete("sys", "p", Answer)
    fake = FakeBatches(lambda _c, _p: succeeded('{"value": 1, "why": "x"}'), polls=0)
    model = BatchedWorkflowModel(Batcher(fake.client(), poll_s=0).lane(), MODEL, effort="low")
    await model.complete("sys", "p", Answer)
    assert fake.batches[0][0]["params"] == bodies[0]
    assert "fallbacks" not in json.dumps(fake.batches[0])
    # server-side refusal fallbacks (F10) are rejected on Batches; SDK request options are not part of the request
    assert batch_params({"model": MODEL, "fallbacks": "default", "timeout": 5, "extra_body": {"fallbacks": "default", "x": 1}}) == {
        "model": MODEL, "x": 1,
    }


async def test_closing_the_batcher_cancels_the_batch_in_flight() -> None:
    fake = FakeBatches(lambda _c, _p: succeeded("{}"), polls=10**9)
    batcher = Batcher(fake.client(), poll_s=0.001)
    call = asyncio.ensure_future(BatchedWorkflowModel(batcher.lane(), MODEL).complete("sys", "p", Answer))
    while not fake.retrievals:
        await asyncio.sleep(0.001)
    await batcher.aclose()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert fake.cancelled == ["msgbatch_0"]


# --- gate 2: an errored or expired result follows F8's api_error path, for that run only -------------------------------

@pytest.mark.parametrize(("result", "kind"), [(errored(), "batch_errored:invalid_request_error"), (expired(), "batch_expired")])
async def test_a_failed_result_ends_only_its_own_run_with_api_error(
    fixture_venv: Path, tmp_path: Path, result: dict[str, Any], kind: str,
) -> None:
    fake = FakeBatches(scripted(replies(), fail={"run2-2": result}))
    batcher = Batcher(fake.client(), flush_s=60, poll_s=0)
    ledger = tmp_path / "bench.jsonl"
    code = await run_grid(
        two_katas(tmp_path), [batched_cell(batcher, cfg(tmp_path, fixture_venv))], sandbox=LocalSandbox(), repeat=2,
        max_cost=1.0, ledger=ledger, parallel=4, out=io.StringIO(), err=io.StringIO(),
    )
    assert code == 0
    assert [len(b) for b in fake.batches] == [4, 4, 3]  # run 2 stopped at its second call; the others went on
    lines = bench.read_ledger(ledger)
    statuses = sorted(x["summary"]["status"] for x in lines)
    assert statuses == ["api_error", "kept", "kept", "kept"]
    failed = next(x["summary"] for x in lines if x["summary"]["status"] == "api_error")
    last = read_log(Path(failed["run_log"]))[-1]
    assert (last["node"], last["at"], last["error"]) == ("api_error", "write_impl", kind)
    assert last["request_id"] in ("req_err", "msgbatch_1/run2-2")
    assert bench.api_errors(lines, "batched") == "batched: 1 run(s) ended with api_error, left out of the means"


async def test_a_failed_batch_request_fails_every_call_in_it() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})

    client = anthropic.AsyncAnthropic(api_key="k", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    batcher = Batcher(client, poll_s=0)
    models = [BatchedWorkflowModel(batcher.lane(), MODEL) for _ in range(2)]
    results = await asyncio.gather(*(m.complete("sys", "p", Answer) for m in models), return_exceptions=True)
    assert all(isinstance(r, ApiFailure) and r.status_code == 500 for r in results)


# --- gate 3: cost at batch prices ---------------------------------------------------------------------------------------

async def test_cost_is_recorded_at_batch_prices() -> None:
    fake = FakeBatches(lambda _c, _p: succeeded('{"value": 1, "why": "x"}', USAGE), polls=0)
    model = BatchedWorkflowModel(Batcher(fake.client(), poll_s=0).lane(), MODEL)
    await model.complete("sys", "p", Answer)
    standard = api_call(MODEL, 1000, 200, 0, 0, None).cost_usd
    assert standard == pytest.approx(0.0002)
    assert model.cost_usd == pytest.approx(standard * BATCH_DISCOUNT) == pytest.approx(0.0001)
    assert model.calls[0].request_id == "msgbatch_0/run1-1"


async def test_the_spend_cap_counts_batch_prices(fixture_venv: Path, tmp_path: Path) -> None:
    # three calls: $0.0003 at batch prices, under the $0.00045 cap; at standard prices ($0.0006) the run would stop
    fake = FakeBatches(scripted(replies()), polls=0)
    batcher = Batcher(fake.client(), poll_s=0)
    ledger = tmp_path / "bench.jsonl"
    config = cfg(tmp_path, fixture_venv, max_cost_usd=0.00045)
    code = await run_grid(
        two_katas(tmp_path), [batched_cell(batcher, config)], sandbox=LocalSandbox(), repeat=1, max_cost=1.0,
        ledger=ledger, out=io.StringIO(), err=io.StringIO(),
    )
    assert code == 0
    summaries = [x["summary"] for x in bench.read_ledger(ledger)]
    assert [s["status"] for s in summaries] == ["kept", "kept"]
    assert all(s["totals"]["cost_usd"] == pytest.approx(0.0003) for s in summaries)


# --- gate 4: the sweep writes one label per cell, and the frontier report reads them ----------------------------------

def bench_args(*argv: str) -> Any:
    return build_parser().parse_args(["bench", "claude-plans", *argv])


def scripted_maker(config: HarnessConfig, choice: Choice) -> Callable[[], WorkflowModel]:
    return lambda: ScriptedModel(replies())


async def test_the_sweep_writes_a_label_per_cell_and_the_frontier_reads_them(fixture_venv: Path, tmp_path: Path) -> None:
    cells = grid_cells(bench_args("--models", MODEL, "--effort", "low,high", "--label", "sw"), cfg(tmp_path, fixture_venv),
                       scripted_maker)
    assert [(c.label, c.config.workflow.effort) for c in cells] == [(f"sw-{MODEL}-low", "low"), (f"sw-{MODEL}-high", "high")]
    ledger = tmp_path / "bench.jsonl"
    out = io.StringIO()
    code = await run_grid(two_katas(tmp_path), cells, sandbox=LocalSandbox(), repeat=1, max_cost=1.0, ledger=ledger,
                          out=out, err=io.StringIO())
    assert code == 0, out.getvalue()
    lines = bench.read_ledger(ledger)
    assert [(x["label"], x["summary"]["effort"]) for x in lines] == [
        (f"sw-{MODEL}-low", "low"), (f"sw-{MODEL}-low", "low"), (f"sw-{MODEL}-high", "high"), (f"sw-{MODEL}-high", "high"),
    ]
    assert f"[sw-{MODEL}-low] alpha r1/1" in out.getvalue()
    text = io.StringIO()
    args = build_parser().parse_args(["report", "--frontier", f"sw-{MODEL}-low", f"sw-{MODEL}-high", "--ledger", str(ledger)])
    assert report(args, HarnessConfig(), text) == 0
    table = text.getvalue()
    assert f"| sw-{MODEL}-low | {MODEL} | low | 2 | - |" in table and f"| sw-{MODEL}-high | {MODEL} | high | 2 | - |" in table
    assert "| alpha |" in table and "| beta |" in table


def test_a_sweep_fails_up_front_on_a_model_without_a_price(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no pricing for 'claude-unpriced-9'"):
        grid_cells(bench_args("--models", f"{MODEL},claude-unpriced-9"), HarnessConfig(), scripted_maker)
    with pytest.raises(ValueError, match="unknown effort turbo"):
        grid_cells(bench_args("--effort", "low,turbo"), HarnessConfig(), scripted_maker)
    with pytest.raises(ValueError, match="ADR 0007"):
        grid_cells(bench_args("--models", "gemma4:e4b"), HarnessConfig(), scripted_maker)
    [plain] = grid_cells(bench_args("--model", MODEL, "--label", "base"), HarnessConfig(), scripted_maker)
    assert plain.label == "base"  # no grid: the label as given, as before F12


def test_the_frontier_marks_the_labels_no_other_beats_on_both() -> None:
    def graded(label: str, model: str, effort: str, kata: str, passed: int, cost: float) -> dict[str, Any]:
        return line(label, kata, 1, 6, cost=cost, model=model, effort=effort, hidden_passed=passed, hidden_total=10,
                    mutants_killed=3, mutants_total=5)

    lines = [
        graded("cheap", "claude-haiku-5-5", "low", "k1", 5, 0.01), graded("cheap", "claude-haiku-5-5", "low", "k2", 5, 0.01),
        graded("mid", "claude-sonnet-5-5", "low", "k1", 8, 0.05), graded("mid", "claude-sonnet-5-5", "low", "k2", 8, 0.05),
        graded("waste", "claude-opus-5-5", "medium", "k1", 7, 0.20), graded("waste", "claude-opus-5-5", "medium", "k2", 7, 0.20),
    ]
    points = {p.label: p for p in bench.frontier_points(lines, ["waste", "cheap", "mid"])}
    assert points["mid"].hidden == pytest.approx(0.8) and points["mid"].cost == pytest.approx(0.05)
    assert [bench.on_frontier(points[x], list(points.values())) for x in ("cheap", "mid", "waste")] == [True, True, False]
    table = bench.frontier_table(lines, ["waste", "cheap", "mid", "nope"])
    rows = [r for r in table.splitlines() if r.startswith("| ") and "claude-" in r]
    assert [r.split(" | ")[0] for r in rows] == ["| cheap", "| mid", "| waste"]  # cheapest first
    assert rows[0].endswith("| 50% | $0.0100 | * |") and rows[2].endswith("| 70% | $0.2000 |  |")
    assert "| k1 | 70% · $0.2000 | 50% · $0.0100 | 80% · $0.0500 |" in table
    assert "no ledger lines labelled 'nope'" in table
