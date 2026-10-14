"""Spend ceilings for runs a client starts over MCP (F18, ADR 0013).

Two ceilings clamp what a client may ask for: ``[mcp_server] max_cost_usd`` per started run and ``daily_cost_usd`` across
every MCP-started run today. The clamped figure becomes the run's Spend cap, which the Workflow machinery then enforces
(``WorkflowConfig.max_cost_usd``); this module only decides the figure and keeps nothing of its own. The daily total comes
from the run index (``RunStore``), where a running run counts at its cap so that concurrent grants cannot overshoot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from harness.workflow.runs import RunStore

ORIGIN = "mcp"  # RunInfo.origin of every run `braid-mcp` starts
EPSILON = 1e-9  # below this much left, a ceiling counts as used up


class SpendLedger(Protocol):
    def spent_today(self, *, excluding: str | None = None) -> float:
        """What MCP-started runs have cost or may still cost since local midnight, without run ``excluding``."""
        ...


class StoreLedger:
    """The daily total, read from the run index."""

    def __init__(self, store: RunStore, now: Callable[[], datetime] = datetime.now) -> None:
        self.store, self.now = store, now

    def spent_today(self, *, excluding: str | None = None) -> float:
        midnight = self.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        since = midnight.astimezone(UTC).isoformat(timespec="microseconds")
        return self.store.spent_since(since, origin=ORIGIN, excluding=excluding)


@dataclass(frozen=True)
class Grant:
    """What a run may spend: ``cap_usd`` is its Spend cap. ``clamped_by`` names the ceiling that lowered the request."""

    cap_usd: float
    requested_usd: float
    clamped_by: Literal["per_run", "daily"] | None
    daily_left_usd: float  # before this grant

    def describe(self) -> str:
        if self.clamped_by is None:
            return f"Spend cap ${self.cap_usd:.4f}"
        ceiling = "the per-run ceiling" if self.clamped_by == "per_run" else "what is left of the daily ceiling"
        return f"Spend cap ${self.cap_usd:.4f} (${self.requested_usd:.4f} was asked; clamped to {ceiling})"

    def fields(self) -> dict[str, object]:
        return {"cap_usd": self.cap_usd, "requested_usd": self.requested_usd, "clamped_by": self.clamped_by,
                "daily_left_usd": round(self.daily_left_usd, 6)}


@dataclass(frozen=True)
class Refusal:
    reason: str


class SpendGuard:
    def __init__(self, ceiling_per_run: float, ceiling_per_day: float, ledger: SpendLedger) -> None:
        self.per_run, self.per_day, self.ledger = ceiling_per_run, ceiling_per_day, ledger

    def clamp(self, requested: float, *, excluding: str | None = None) -> Grant | Refusal:
        """The Spend cap for a run that asks for ``requested`` dollars: the smallest of that, the per-run ceiling and what
        is left of the daily one. A refusal when the request is not a positive number or the daily ceiling is used up.
        ``excluding`` is a run already counted in the ledger that is asking for more (``braid_run_answer``)."""
        if not requested > 0:
            return Refusal(f"max_cost_usd must be more than zero, not {requested}")
        spent = self.ledger.spent_today(excluding=excluding)
        left = self.per_day - spent
        if left <= EPSILON:
            return Refusal(f"the daily ceiling is used up: ${spent:.4f} of ${self.per_day:.2f} spent today by MCP-started runs "
                           f"(raise [mcp_server] daily_cost_usd, or start runs from the TUI)")
        cap = requested
        by: Literal["per_run", "daily"] | None = None
        if self.per_run < cap:
            cap, by = self.per_run, "per_run"
        if left < cap:
            cap, by = left, "daily"
        return Grant(round(cap, 6), requested, by, left)
