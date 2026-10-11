"""M9 gate: one command produces a per-model comparison of accuracy, tokens per second and peak context use.

Fakes only: a fake Ollama backend and fake agents, no network. The Strands Evals tests skip when the optional
``evals`` group is not installed; the aggregation tests do not need it.
"""

import importlib.util
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pytest

from harness.evals.cli import main
from harness.evals.report import CallLog, CaseRun, ModelResult, render_markdown, write_reports
from harness.evals.runner import EvalRuntime, load_prompts
from harness.guard.ollama_client import CallMetrics
from tests.guard.fakes import GIB, FakeBackend

HAS_EVALS = importlib.util.find_spec("strands_evals") is not None
needs_evals = pytest.mark.skipif(not HAS_EVALS, reason="optional `evals` dependency group not installed")

PROMPTS: list[dict[str, Any]] = [
    {"prompt": "What time is it?", "tool": "current_time"},
    {"prompt": "List src", "tool": "list_directory"},
    {"prompt": "Write a haiku", "tool": None},
    {"prompt": "Read README.md", "tool": "read_file"},
]


def call(prompt: int, eval_count: int = 50, secs: float = 1.0) -> CallMetrics:
    return CallMetrics(prompt_eval_count=prompt, eval_count=eval_count, eval_duration_s=secs, load_duration_s=0.0)


# -- aggregation and report: no Strands Evals needed ---------------------------------------------------------

def result(model: str, flags: list[bool], num_ctx: int = 1000) -> ModelResult:
    runs = [
        CaseRun(f"{i:02d}", f"p{i}", "t", "t" if ok else None, ok, 2.0, (call(100 * (i + 1), 40, 1.0), call(50, 60, 2.0)))
        for i, ok in enumerate(flags)
    ]
    return ModelResult(model, num_ctx=num_ctx, cases=runs)


def test_model_result_aggregates_accuracy_speed_and_context() -> None:
    r = result("a", [True, True, False, True])
    assert (r.correct, r.n, r.accuracy) == (3, 4, 0.75)
    assert sorted(r.tps_values) == [30.0] * 4 + [40.0] * 4
    assert r.mean_tps == 35.0 and r.median_tps == 35.0
    assert r.peak_prompt_tokens == 400 and r.peak_context_share == 0.4
    assert r.mean_prompt_tokens == (100 + 200 + 300 + 400 + 4 * 50) / 8
    assert r.wall_s == 8.0 and r.errors == 0


def test_empty_result_has_no_metrics_instead_of_crashing() -> None:
    r = ModelResult("x", num_ctx=1000)
    assert r.accuracy is None and r.mean_tps is None and r.peak_context_share is None


def test_markdown_and_json_reports_are_written(tmp_path: Path) -> None:
    results = [result("good", [True] * 4), result("bad", [False] * 4), ModelResult("gone", status="skipped", reason="not pulled")]
    when = datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC)
    md_path, json_path, md = write_reports(results, tmp_path / "evals", num_ctx=1000, cases_path="x.json", when=when)
    assert md_path.name == "20261008T120000Z.md" and json_path.name == "20261008T120000Z.json"
    assert md_path.read_text() == md
    assert "| good | 4/4 (100%) |" in md and "| bad | 0/4 (0%) |" in md and "40.0% of 1000" in md
    assert "- gone: skipped, not pulled" in md
    doc = json.loads(json_path.read_text())
    assert [m["model"] for m in doc["models"]] == ["good", "bad", "gone"]
    assert doc["models"][0]["accuracy"] == 1.0 and len(doc["models"][0]["case_runs"]) == 4
    assert render_markdown(results, when=when, num_ctx=1000, cases_path="x.json") == md


def test_default_cases_file_is_the_m4_set() -> None:
    from harness.evals.runner import DEFAULT_CASES

    assert len(load_prompts(DEFAULT_CASES)) == 20 and len(load_prompts(DEFAULT_CASES, limit=3)) == 3


# -- the one command, end to end with fakes -------------------------------------------------------------------

class FakeAgent:
    """Calls ``tool`` (or none) like a Strands agent would, and logs one measured call like the guard client."""

    def __init__(self, tool: str | None, log: CallLog, metrics: CallMetrics) -> None:
        self.messages: list[Any] = []
        self._tool, self._log, self._metrics = tool, log, metrics

    async def invoke_async(self, prompt: str) -> str:
        self._log.add(self._metrics)
        if self._tool:
            self.messages.append({"role": "assistant", "content": [{"toolUse": {"name": self._tool, "input": {}}}]})
        return "done"


def runtime() -> tuple[EvalRuntime, FakeBackend, list[str]]:
    backend = FakeBackend(models={"smart": (2 * GIB, 1000.0), "dumb": (2 * GIB, 1000.0)})
    labels = {p["prompt"]: p["tool"] for p in PROMPTS}
    built: list[str] = []

    def make_factory(model: str, log: CallLog) -> Any:
        built.append(model)
        metrics = call(300, 100, 2.0) if model == "smart" else call(900, 40, 2.0)

        def factory() -> Any:
            agent = FakeAgent(None, log, metrics)
            original = agent.invoke_async

            async def invoke(prompt: str) -> str:
                if model == "smart":
                    agent._tool = labels[prompt]  # always right
                else:
                    agent._tool = "web_search"  # never right (and "calls a tool" even when none was wanted)
                return await original(prompt)

            agent.invoke_async = invoke  # type: ignore[method-assign]
            return agent

        return factory

    return EvalRuntime(backend, make_factory, {"OLLAMA_NUM_PARALLEL": "1"}, gpu_limit=100 * GIB), backend, built


@needs_evals
def test_eval_command_compares_models_and_lists_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(PROMPTS))
    rt, backend, _ = runtime()
    cfg_file = tmp_path / "h.toml"
    cfg_file.write_text(f'[paths]\ndata_dir = "{tmp_path / "data"}"\n[evals]\nnum_ctx = 2000\n')

    main(["--config", str(cfg_file), "--models", "smart,dumb,missing", "--cases", str(cases)], runtime_factory=lambda _c: rt)

    out = capsys.readouterr().out
    assert "| smart | 4/4 (100%) | 50.0 | 50.0 | 300 | 15.0% of 2000 |" in out
    assert "| dumb | 0/4 (0%) | 20.0 | 20.0 | 900 | 45.0% of 2000 |" in out
    assert "- missing: skipped, not pulled" in out
    evals_dir = tmp_path / "data" / "evals"
    (md,) = evals_dir.glob("*.md")
    (js,) = evals_dir.glob("*.json")
    assert md.read_text().strip() in out
    models = {m["model"]: m for m in json.loads(js.read_text())["models"]}
    assert models["smart"]["accuracy"] == 1.0 and models["dumb"]["accuracy"] == 0.0
    assert models["smart"]["peak_prompt_tokens"] == 300 and models["dumb"]["mean_tokens_per_second"] == 20.0
    assert models["missing"]["status"] == "skipped"
    assert backend.loaded == {}, "every model is unloaded after its run"


@needs_evals
async def test_run_model_counts_an_erroring_agent_as_wrong_not_as_a_crash() -> None:
    from harness.evals.runner import run_model

    class Boom:
        messages: list[Any] = []

        async def invoke_async(self, prompt: str) -> str:
            raise RuntimeError("ollama: out of memory")

    r = await run_model("m", PROMPTS[:2], lambda: Boom(), num_ctx=1000, log=CallLog())
    assert (r.correct, r.errors) == (0, 2) and "out of memory" in r.cases[0].error


@needs_evals
async def test_no_tool_label_passes_only_when_no_tool_is_called() -> None:
    from harness.evals.runner import run_model

    log = CallLog()
    r = await run_model("m", [PROMPTS[2]], lambda: FakeAgent(None, log, call(10)), num_ctx=100, log=log)
    assert r.correct == 1
    r = await run_model("m", [PROMPTS[2]], lambda: FakeAgent("current_time", log, call(10)), num_ctx=100, log=log)
    assert r.correct == 0 and r.cases[0].first_tool == "current_time"
