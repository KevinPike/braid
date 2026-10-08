"""Harness hooks: every Guard, context, decision, session and approval event goes through one Strands registry.

Components keep their own ``subscribe`` methods (``guard/`` must not import Strands, ADR 0001). The
adapters here subscribe to each component exactly once and republish its events as Strands hook
events, so the TUI and any later consumer (M9's OpenTelemetry provider) register on
``HarnessHooks.registry`` the same way instead of reaching into components.

Events are published synchronously, on the thread and loop that raised them, exactly as the
components' own listeners were called before; callbacks must therefore be plain functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strands.hooks import BaseHookEvent, HookCallback, HookProvider, HookRegistry
from strands.hooks.registry import TEvent

from harness.context.ledger import CallRecord, Ledger
from harness.context.trimlog import TrimEntry, TrimLog
from harness.decide.log import Decision, DecisionLog
from harness.guard.ollama_client import CallMetrics, OllamaClient
from harness.guard.watchdog import GuardState, Watchdog
from harness.prompts import PromptQueue

if TYPE_CHECKING:  # approval.py imports these events, so the request type is only named, never imported, here
    from harness.approval import ApprovalRequest


@dataclass
class GuardStateChanged(BaseHookEvent):
    """The Watchdog published a new state (a poll or a completed call)."""

    state: GuardState


@dataclass
class CallRecorded(BaseHookEvent):
    """The Ledger recorded a model call: first the estimate (in flight), then again once measured."""

    record: CallRecord


@dataclass
class TrimRecorded(BaseHookEvent):
    entry: TrimEntry


@dataclass
class DecisionRecorded(BaseHookEvent):
    decision: Decision


@dataclass
class ModelRequestSent(BaseHookEvent):
    """The guard client is about to POST ``body`` to ``/api/chat``."""

    model: str
    body: dict[str, Any]


@dataclass
class ModelCallCompleted(BaseHookEvent):
    """A call to the guard client finished with measured counts; ``estimate`` is the caller's, if it had one."""

    model: str
    metrics: CallMetrics
    estimate: int | None


@dataclass
class QueueChanged(BaseHookEvent):
    """The Prompt queue's pending prompts changed."""

    pending: tuple[str, ...]


@dataclass
class ApprovalRequested(BaseHookEvent):
    request: ApprovalRequest


@dataclass
class ApprovalResolved(BaseHookEvent):
    request: ApprovalRequest
    approved: bool


class HarnessHooks:
    """Owns the registry for harness events and adapts each component's listener onto it."""

    def __init__(self) -> None:
        self.registry = HookRegistry()
        self._attached: set[int] = set()

    def add_callback(self, event_type: type[TEvent], callback: HookCallback[TEvent]) -> None:
        self.registry.add_callback(event_type, callback)

    def add_hook(self, provider: HookProvider) -> None:
        self.registry.add_hook(provider)

    def publish(self, event: BaseHookEvent) -> None:
        """Call every callback registered for the event's type, in registration order, synchronously."""
        self.registry.invoke_callbacks(event)

    def _once(self, component: object) -> bool:
        if id(component) in self._attached:
            return False
        self._attached.add(id(component))
        return True

    def attach_watchdog(self, watchdog: Watchdog) -> None:
        if self._once(watchdog):
            watchdog.subscribe(lambda state: self.publish(GuardStateChanged(state)))

    def attach_ledger(self, ledger: Ledger) -> None:
        if self._once(ledger):
            ledger.subscribe(lambda record: self.publish(CallRecorded(record)))

    def attach_trimlog(self, trimlog: TrimLog) -> None:
        if self._once(trimlog):
            trimlog.subscribe(lambda entry: self.publish(TrimRecorded(entry)))

    def attach_decisions(self, decisions: DecisionLog) -> None:
        if self._once(decisions):
            decisions.subscribe(lambda decision: self.publish(DecisionRecorded(decision)))

    def attach_client(self, client: OllamaClient) -> None:
        if self._once(client):
            client.on_request(lambda model, body: self.publish(ModelRequestSent(model, body)))
            client.on_call(lambda model, metrics, estimate: self.publish(ModelCallCompleted(model, metrics, estimate)))

    def attach_queue(self, queue: PromptQueue) -> None:
        if self._once(queue):
            queue.subscribe(lambda pending: self.publish(QueueChanged(pending)))


__all__ = [
    "ApprovalRequested", "ApprovalResolved", "CallRecorded", "DecisionRecorded", "GuardStateChanged",
    "HarnessHooks", "ModelCallCompleted", "ModelRequestSent", "QueueChanged", "TrimRecorded",
]
