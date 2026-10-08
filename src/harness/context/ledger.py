"""Ledger: one record per model call, comparing the harness estimate with what Ollama measured."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness.context.accounting import Accounting, Category, TokenEstimator, account
from harness.guard.ollama_client import CallMetrics
from harness.guard.watchdog import TRUNCATION_TOLERANCE

log = logging.getLogger(__name__)

# A prompt estimated below this share of num_ctx cannot have been truncated, so it is safe to learn from.
LEARN_BELOW_CTX_SHARE = 0.8


@dataclass(frozen=True)
class CallRecord:
    num_ctx: int
    categories: Mapping[Category, int]  # estimated tokens per category
    estimate: int
    strands_estimate: int | None
    prompt_eval_count: int | None = None  # filled when the call completes
    eval_count: int | None = None

    @property
    def measured(self) -> bool:
        return self.prompt_eval_count is not None

    @property
    def gap(self) -> int | None:
        """Estimate minus measured prompt tokens; large and positive means Truncation."""
        return None if self.prompt_eval_count is None else self.estimate - self.prompt_eval_count

    @property
    def error(self) -> float | None:
        if self.prompt_eval_count is None or self.prompt_eval_count == 0:
            return None
        return abs(self.estimate - self.prompt_eval_count) / self.prompt_eval_count

    def scaled_categories(self) -> Mapping[Category, int]:
        """Category tokens rescaled to sum to the measured total (identity before the call completes)."""
        if self.prompt_eval_count is None or self.estimate == 0:
            return self.categories
        k = self.prompt_eval_count / self.estimate
        return {c: round(n * k) for c, n in self.categories.items()}


class Ledger:
    def __init__(self, num_ctx: Callable[[str], int | None], estimator: TokenEstimator | None = None) -> None:
        self._num_ctx = num_ctx
        self.estimator = estimator or TokenEstimator()
        self.records: list[CallRecord] = []
        self._pending: tuple[Accounting, CallRecord] | None = None
        self._listeners: list[Callable[[CallRecord], None]] = []

    def subscribe(self, listener: Callable[[CallRecord], None]) -> None:
        self._listeners.append(listener)

    def before_call(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        strands_estimate: int | None = None,
    ) -> int:
        """Account the request about to be sent; returns the total token estimate."""
        acct = account(messages, tools)
        cats = self.estimator.estimate(acct)
        record = CallRecord(self._num_ctx(model) or 0, cats, sum(cats.values()), strands_estimate)
        self._pending = (acct, record)
        self._emit(record)
        return record.estimate

    def after_call(self, metrics: CallMetrics) -> CallRecord | None:
        if self._pending is None:
            return None
        acct, rec = self._pending
        self._pending = None
        done = CallRecord(
            rec.num_ctx, rec.categories, rec.estimate, rec.strands_estimate,
            metrics.prompt_eval_count, metrics.eval_count,
        )
        if rec.num_ctx and rec.estimate < LEARN_BELOW_CTX_SHARE * rec.num_ctx:
            self.estimator.observe(acct, metrics.prompt_eval_count)
        if done.gap is not None and rec.num_ctx and metrics.prompt_eval_count < rec.estimate * (1 - TRUNCATION_TOLERANCE):
            log.warning(
                "truncation: estimate %d, evaluated %d, gap %d tokens (num_ctx %d)",
                rec.estimate, metrics.prompt_eval_count, done.gap, rec.num_ctx,
            )
        self.records.append(done)
        self._emit(done)
        return done

    def _emit(self, record: CallRecord) -> None:
        for listener in self._listeners:
            listener(record)
