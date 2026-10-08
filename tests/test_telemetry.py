"""M9 telemetry: harness events become OpenTelemetry metrics; nothing is installed when disabled. No network."""

from collections.abc import Mapping
from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness.approval import Advice, ApprovalRequest
from harness.config import HarnessConfig, TelemetryConfig, load_config
from harness.context.ledger import CallRecord
from harness.context.trimlog import TrimEntry
from harness.decide.log import Decision
from harness.guard.ollama_client import CallMetrics
from harness.guard.watchdog import Alert, GuardState, Level
from harness.hooks import (
    ApprovalResolved,
    CallRecorded,
    DecisionRecorded,
    GuardStateChanged,
    HarnessHooks,
    ModelCallCompleted,
    TrimRecorded,
)
from harness.telemetry import setup_telemetry

ON = TelemetryConfig(enabled=True)


def installed() -> tuple[HarnessHooks, InMemoryMetricReader, InMemorySpanExporter]:  # spans: injected, never exported over a network
    reader = InMemoryMetricReader()
    spans = InMemorySpanExporter()
    tracer = TracerProvider()
    tracer.add_span_processor(SimpleSpanProcessor(spans))
    hooks = HarnessHooks()
    assert setup_telemetry(ON, hooks, tracer_provider=tracer, meter_provider=MeterProvider(metric_readers=[reader])) is not None
    return hooks, reader, spans


def points(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    data = reader.get_metrics_data()
    out: dict[str, list[Any]] = {}
    for rm in (data.resource_metrics if data else []):
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out.setdefault(metric.name, []).extend(metric.data.data_points)
    return out


def attrs(point: Any) -> Mapping[str, Any]:
    return dict(point.attributes)


def measured(model: str = "m1", num_ctx: int = 1000, estimate: int = 120, prompt: int = 100, eval_count: int = 20) -> CallRecord:
    return CallRecord(num_ctx, {}, estimate, None, prompt, eval_count, model)


def test_call_metrics_carry_the_model_and_skip_the_in_flight_estimate() -> None:
    hooks, reader, _ = installed()
    hooks.publish(CallRecorded(CallRecord(1000, {}, 120, None, model="m1")))  # estimate only: not a data point
    hooks.publish(CallRecorded(measured()))
    hooks.publish(ModelCallCompleted("m1", CallMetrics(100, 20, 0.5, 0.0), 120))
    p = points(reader)
    assert len(p["braid.call.prompt_tokens"]) == 1
    assert p["braid.call.prompt_tokens"][0].sum == 100 and attrs(p["braid.call.prompt_tokens"][0]) == {"model": "m1"}
    assert p["braid.call.prompt_tokens_estimate"][0].sum == 120
    assert p["braid.call.eval_tokens"][0].sum == 20
    assert p["braid.call.context_use"][0].sum == 0.1
    assert abs(p["braid.call.estimate_error"][0].sum - 0.2) < 1e-9
    assert p["braid.call.tokens_per_second"][0].sum == 40.0
    assert attrs(p["braid.call.tokens_per_second"][0]) == {"model": "m1"}
    assert p["braid.calls"][0].value == 1


def test_trims_decisions_and_approvals_are_counted_by_label() -> None:
    hooks, reader, _ = installed()
    hooks.publish(TrimRecorded(TrimEntry("summarize", "ceiling", 4, 900, 300)))
    hooks.publish(TrimRecorded(TrimEntry("summarize", "ceiling", 2, 500, 400)))
    hooks.publish(TrimRecorded(TrimEntry("offload", "big result", 0, 5000, 100)))
    hooks.publish(DecisionRecorded(Decision("router", "tev1:0.8b", "s", {})))
    hooks.publish(DecisionRecorded(Decision("policy", "tev1:0.8b", "s", {}, error="down")))
    req = ApprovalRequest("shell", {"command": "ls"}, Advice())
    hooks.publish(ApprovalResolved(req, True))
    hooks.publish(ApprovalResolved(req, False))
    hooks.publish(ApprovalResolved(req, False))
    p = points(reader)
    trims = {attrs(x)["strategy"]: x.value for x in p["braid.trims"]}
    assert trims == {"summarize": 2, "offload": 1}
    freed = {attrs(x)["strategy"]: x.value for x in p["braid.trim.tokens_freed"]}
    assert freed == {"summarize": 700, "offload": 4900}
    decisions = {(attrs(x)["kind"], attrs(x)["outcome"]): x.value for x in p["braid.decisions"]}
    assert decisions == {("router", "ok"): 1, ("policy", "error"): 1}
    approvals = {attrs(x)["outcome"]: x.value for x in p["braid.approvals"]}
    assert approvals == {"approved": 1, "denied": 2}


def test_alert_counter_counts_transitions_not_polls() -> None:
    hooks, reader, _ = installed()
    spill = Alert("spill", Level.RED, "spilled")
    drift = Alert("num_ctx_drift", Level.RED, "drift")
    for alerts in [(), (spill,), (spill,), (spill,), (spill, drift), (drift,), (), (spill,)]:
        hooks.publish(GuardStateChanged(GuardState(model="m1", alerts=alerts)))
    counts = {(attrs(x)["kind"], attrs(x)["level"]): x.value for x in points(reader)["braid.guard.alerts"]}
    assert counts == {("spill", "red"): 2, ("num_ctx_drift", "red"): 1}
    assert all(attrs(x)["model"] == "m1" for x in points(reader)["braid.guard.alerts"])


def test_disabled_installs_nothing() -> None:
    hooks = HarnessHooks()
    assert setup_telemetry(TelemetryConfig(), hooks) is None
    assert not hooks.registry.has_callbacks()


def test_telemetry_config_defaults_and_toml(tmp_path: Any) -> None:
    cfg = HarnessConfig().telemetry
    assert (cfg.enabled, cfg.endpoint, cfg.service_name) == (False, "http://localhost:4318", "braid")
    f = tmp_path / "h.toml"
    f.write_text('[telemetry]\nenabled = true\nservice_name = "x"\n[evals]\nmodels = ["a"]\n')
    loaded = load_config(f)
    assert loaded.telemetry.enabled and loaded.telemetry.service_name == "x" and loaded.evals.models == ["a"]
    assert loaded.paths.evals_dir == loaded.paths.resolved_data_dir / "evals"

