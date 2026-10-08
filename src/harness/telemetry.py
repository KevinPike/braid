"""OpenTelemetry for the harness (M9): Strands' agent, model and tool spans plus metrics from harness events.

Off by default (``[telemetry] enabled = false``); with it off nothing here runs and nothing is registered.

Run a local collector, then set ``enabled = true`` in ``harness.toml``::

    # spans and metrics printed by the collector (debug exporter)
    docker run --rm -p 4318:4318 otel/opentelemetry-collector

    # or Jaeger for looking at traces (UI at http://localhost:16686)
    docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one

Traces come from Strands itself (``StrandsTelemetry`` installs the tracer provider and the OTLP/HTTP span
exporter). Metrics are produced by ``MetricsHook``, a Strands ``HookProvider`` registered on
``app.hooks``, so it sees the same events as the TUI: calls (tokens, tokens per second, context use,
estimate error), trims, decisions, approvals, and Guard alerts counted when they appear, not on every poll.

The exporters never get in the UI's way. Spans go through a ``BatchSpanProcessor`` and metrics through a
``PeriodicExportingMetricReader``; both export from their own background threads with a short timeout. When
no collector is listening the SDK logs the failure from that thread and drops the data; nothing is raised
into the harness. Those SDK log lines are kept off stderr (they would scribble over the TUI) by a
``NullHandler`` on the ``opentelemetry`` logger.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from opentelemetry import metrics as metrics_api
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from strands.hooks import HookProvider, HookRegistry

from harness.config import TelemetryConfig
from harness.hooks import (
    ApprovalResolved,
    CallRecorded,
    DecisionRecorded,
    GuardStateChanged,
    HarnessHooks,
    ModelCallCompleted,
    TrimRecorded,
)

EXPORT_TIMEOUT_S = 3.0  # per OTLP request; the exporters run on background threads either way

Attributes = dict[str, str | int]


class MetricsHook(HookProvider):
    """Turns harness events into OpenTelemetry instruments."""

    def __init__(self, meter: Meter) -> None:
        self._prompt_estimate = meter.create_histogram(
            "braid.call.prompt_tokens_estimate", unit="{token}", description="Harness estimate of the prompt, per call")
        self._prompt_tokens = meter.create_histogram(
            "braid.call.prompt_tokens", unit="{token}", description="prompt_eval_count Ollama measured, per call")
        self._eval_tokens = meter.create_histogram(
            "braid.call.eval_tokens", unit="{token}", description="Generated tokens, per call")
        self._tps = meter.create_histogram(
            "braid.call.tokens_per_second", unit="{token}/s", description="Generation speed, per call")
        self._context_use = meter.create_histogram(
            "braid.call.context_use", unit="1", description="prompt_eval_count / num_ctx, per call")
        self._estimate_error = meter.create_histogram(
            "braid.call.estimate_error", unit="1", description="|estimate - measured| / measured, per call")
        self._calls = meter.create_counter("braid.calls", unit="{call}", description="Completed model calls")
        self._trims = meter.create_counter("braid.trims", unit="{trim}", description="Context trims by strategy")
        self._tokens_freed = meter.create_counter(
            "braid.trim.tokens_freed", unit="{token}", description="Tokens freed by trims, by strategy")
        self._decisions = meter.create_counter("braid.decisions", unit="{decision}", description="Decision layer calls by kind")
        self._approvals = meter.create_counter("braid.approvals", unit="{approval}", description="Approvals by outcome")
        self._alerts = meter.create_counter(
            "braid.guard.alerts", unit="{alert}", description="Guard alerts, counted when they first appear")
        self._active_alerts: set[tuple[str, str]] = set()

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(CallRecorded, self._on_record)
        registry.add_callback(ModelCallCompleted, self._on_call)
        registry.add_callback(GuardStateChanged, self._on_guard)
        registry.add_callback(TrimRecorded, self._on_trim)
        registry.add_callback(DecisionRecorded, self._on_decision)
        registry.add_callback(ApprovalResolved, self._on_approval)

    def _on_record(self, event: CallRecorded) -> None:
        rec = event.record
        if not rec.measured:  # the in-flight estimate is published first; only the measured record is a data point
            return
        attrs: Attributes = {"model": rec.model}
        assert rec.prompt_eval_count is not None
        self._prompt_estimate.record(rec.estimate, attrs)
        self._prompt_tokens.record(rec.prompt_eval_count, attrs)
        if rec.eval_count is not None:
            self._eval_tokens.record(rec.eval_count, attrs)
        if rec.num_ctx:
            self._context_use.record(rec.prompt_eval_count / rec.num_ctx, attrs)
        if rec.error is not None:
            self._estimate_error.record(rec.error, attrs)

    def _on_call(self, event: ModelCallCompleted) -> None:
        attrs: Attributes = {"model": event.model}
        self._calls.add(1, attrs)
        tps = event.metrics.tokens_per_second
        if tps is not None:
            self._tps.record(tps, attrs)

    def _on_guard(self, event: GuardStateChanged) -> None:
        """Count an alert when its (kind, level) appears, not on each poll that still carries it."""
        now = {(a.kind, a.level.name.lower()) for a in event.state.alerts}
        for kind, level in sorted(now - self._active_alerts):
            self._alerts.add(1, {"kind": kind, "level": level, "model": event.state.model or ""})
        self._active_alerts = now

    def _on_trim(self, event: TrimRecorded) -> None:
        attrs: Attributes = {"strategy": event.entry.strategy}
        self._trims.add(1, attrs)
        self._tokens_freed.add(max(event.entry.tokens_freed, 0), attrs)

    def _on_decision(self, event: DecisionRecorded) -> None:
        d = event.decision
        self._decisions.add(1, {"kind": d.kind, "model": d.model, "outcome": "error" if d.error else "ok"})

    def _on_approval(self, event: ApprovalResolved) -> None:
        self._approvals.add(1, {"tool": event.request.tool, "outcome": "approved" if event.approved else "denied"})


@dataclass
class Telemetry:
    """What ``setup_telemetry`` installed; ``shutdown`` flushes the exporters (also done at interpreter exit)."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def shutdown(self) -> None:
        for provider in (self.tracer_provider, self.meter_provider):
            try:
                provider.shutdown()
            except Exception:  # never let a dead collector turn into a crash on the way out
                logging.getLogger(__name__).debug("telemetry shutdown failed", exc_info=True)


def _quiet_sdk_logging() -> None:
    sdk_log = logging.getLogger("opentelemetry")
    if not any(isinstance(h, logging.NullHandler) for h in sdk_log.handlers):
        sdk_log.addHandler(logging.NullHandler())
    sdk_log.propagate = False  # no stderr fallback while the TUI owns the terminal


def setup_telemetry(
    config: TelemetryConfig,
    hooks: HarnessHooks,
    *,
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
) -> Telemetry | None:
    """Install tracing and metrics when ``config.enabled``; otherwise do nothing at all.

    Passing providers (tests) skips the OTLP exporters and the global registration, so no network is touched.
    """
    if not config.enabled:
        return None
    _quiet_sdk_logging()
    endpoint = config.endpoint.rstrip("/")
    if tracer_provider is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from strands.telemetry.config import StrandsTelemetry

        os.environ.setdefault("OTEL_SERVICE_NAME", config.service_name)
        strands_telemetry = StrandsTelemetry()  # sets the global tracer provider Strands' spans use
        tracer_provider = strands_telemetry.tracer_provider
        add_span_exporter(tracer_provider, OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=EXPORT_TIMEOUT_S))
    if meter_provider is None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", timeout=EXPORT_TIMEOUT_S),
            export_interval_millis=int(config.metric_interval_s * 1000),
            export_timeout_millis=int(EXPORT_TIMEOUT_S * 1000),
        )
        meter_provider = build_meter_provider(config.service_name, [reader])
        metrics_api.set_meter_provider(meter_provider)  # Strands' own event-loop metrics ride along
    hooks.add_hook(MetricsHook(meter_provider.get_meter("braid")))
    return Telemetry(tracer_provider, meter_provider)


def add_span_exporter(provider: TracerProvider, exporter: SpanExporter) -> None:
    provider.add_span_processor(BatchSpanProcessor(exporter))


def build_meter_provider(service_name: str, readers: Iterable[MetricReader]) -> MeterProvider:
    return MeterProvider(resource=Resource.create({"service.name": service_name}), metric_readers=list(readers))
