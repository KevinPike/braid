"""F12 Batches: a Workflow's calls at half price, through the Message Batches API.

A Workflow's calls are sequential, so a batch per call would add a batch's latency to every node. Instead N bench runs
go at once, each through its own ``BatchLane``, and the ``Batcher`` gathers their pending calls into one batch once every
active run is waiting (or ``flush_s`` after the first of them came in): the runs move in lockstep, a node at a time.
Results come back in any order and are matched to their call by ``custom_id``.

A ``BatchLane`` stands in for the ``AsyncAnthropic`` client, so ``BatchedWorkflowModel`` is ``AnthropicWorkflowModel``
unchanged: a batched call carries exactly the params a single call would, keeps its retry-once on invalid JSON, and
fails the same way (``ApiFailure``, F8's ``api_error`` path, for that run only).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any, cast

import anthropic

from harness.anthropic_api import API_FAULTS, ApiFailure
from harness.config import Effort
from harness.usage import ApiCall
from harness.workflow.model import AnthropicWorkflowModel

BATCH_DISCOUNT = 0.5  # Message Batches bill every token at half the standard price

# Request options are the SDK's, not the request's; server-side refusal fallbacks are rejected on Batches.
_REQUEST_OPTIONS = ("extra_headers", "extra_query", "extra_body", "timeout")
_REJECTED = ("fallbacks",)


def batch_priced(call: ApiCall) -> ApiCall:
    """``call`` at batch prices. A helper rather than a ``PRICING`` entry: the discount applies to every model."""
    return replace(call, cost_usd=call.cost_usd * BATCH_DISCOUNT)


def batch_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """What ``messages.create(**params)`` would send, as one batch request's ``params``, less what Batches reject."""
    body = {k: v for k, v in params.items() if k not in _REQUEST_OPTIONS}
    body.update(params.get("extra_body") or {})
    for key in _REJECTED:
        body.pop(key, None)
    return body


Waiter = asyncio.Future[anthropic.types.Message]


class Batcher:
    """Gathers the calls of every open ``BatchLane`` into batches, and hands each result back to its caller.

    A batch is sent once every active lane has a call waiting, or ``flush_s`` after the first waiting call, whichever
    comes first. Then it is polled every ``poll_s`` until it has ended. An errored, expired or canceled result fails
    only its own call, with an ``ApiFailure``; a batch request that fails past the SDK's retries fails every call in it."""

    def __init__(
        self, client: anthropic.AsyncAnthropic, *, flush_s: float = 30.0, poll_s: float = 15.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.flush_s = flush_s
        self.poll_s = poll_s
        self.sleep = sleep
        self.active = 0  # lanes open: runs that will call again
        self.batches: list[list[str]] = []  # the custom_ids sent in each batch, in order
        self._lanes = 0
        self._pending: dict[str, tuple[dict[str, Any], Waiter]] = {}
        self._timer: asyncio.TimerHandle | None = None
        self._flushes: set[asyncio.Task[None]] = set()

    def lane(self) -> BatchLane:
        """A new run's lane; it counts as active until it is closed."""
        self._lanes += 1
        self.active += 1
        return BatchLane(self, f"run{self._lanes}")

    def _leave(self) -> None:
        self.active -= 1
        self._check()  # the runs still waiting may be all that is left

    async def submit(self, custom_id: str, params: dict[str, Any]) -> anthropic.types.Message:
        waiter: Waiter = asyncio.get_running_loop().create_future()
        self._pending[custom_id] = (params, waiter)
        self._check()
        try:
            return await waiter
        finally:
            if self._pending.get(custom_id, (None, None))[1] is waiter:  # cancelled before its batch was sent
                del self._pending[custom_id]
                self._check()

    def _check(self) -> None:
        if not self._pending:
            self._cancel_timer()
        elif len(self._pending) >= self.active:
            self._flush()
        elif self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(self.flush_s, self._flush)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _flush(self) -> None:
        self._cancel_timer()
        calls, self._pending = self._pending, {}
        if calls:
            task = asyncio.ensure_future(self._send(calls))
            self._flushes.add(task)
            task.add_done_callback(self._flushes.discard)

    async def _send(self, calls: dict[str, tuple[dict[str, Any], Waiter]]) -> None:
        self.batches.append(list(calls))
        batches = self.client.messages.batches
        batch_id: str | None = None
        try:
            batch = await batches.create(requests=[
                {"custom_id": cid, "params": cast(Any, params)} for cid, (params, _) in calls.items()
            ])
            batch_id = batch.id
            while batch.processing_status != "ended":
                await self.sleep(self.poll_s)
                batch = await batches.retrieve(batch.id)
            async for item in await batches.results(batch.id):
                _, waiter = calls.pop(item.custom_id, (None, None))
                if waiter is not None and not waiter.done():
                    _resolve(waiter, item.result, batch.id, item.custom_id)
            for cid, (_, waiter) in calls.items():
                if not waiter.done():
                    waiter.set_exception(ApiFailure("MissingResult", f"batch {batch.id} returned no result for {cid}"))
        except API_FAULTS as exc:  # past the SDK's retries: every call in this batch fails (F8)
            for _, waiter in calls.values():
                if not waiter.done():
                    waiter.set_exception(ApiFailure.from_exc(exc))
        except asyncio.CancelledError:
            for _, waiter in calls.values():
                waiter.cancel()
            if batch_id is not None:  # best effort: stop paying for work nobody will read
                with contextlib.suppress(*API_FAULTS):
                    await asyncio.shield(batches.cancel(batch_id))
            raise

    async def aclose(self) -> None:
        """Cancel the batches still in flight (and on the server)."""
        self._cancel_timer()
        for task in list(self._flushes):
            task.cancel()
        await asyncio.gather(*self._flushes, return_exceptions=True)


def _resolve(waiter: Waiter, result: Any, batch_id: str, custom_id: str) -> None:
    """One result for its waiting call: the message, or an ``ApiFailure`` for that call only."""
    if result.type == "succeeded":
        message = result.message
        message._request_id = f"{batch_id}/{custom_id}"  # where this reply came from, in the run log
        waiter.set_result(message)
    elif result.type == "errored":
        error = result.error
        waiter.set_exception(ApiFailure(
            f"batch_errored:{error.error.type}", error.error.message, request_id=error.request_id or f"{batch_id}/{custom_id}",
        ))
    else:  # expired (the batch's 24 hours ran out) or canceled
        waiter.set_exception(ApiFailure(f"batch_{result.type}", f"batch {batch_id} {result.type} before {custom_id} ran",
                                        request_id=f"{batch_id}/{custom_id}"))


class _LaneMessages:
    def __init__(self, lane: BatchLane) -> None:
        self.lane = lane

    async def create(self, **params: Any) -> anthropic.types.Message:
        lane = self.lane
        lane.calls += 1
        return await lane.batcher.submit(f"{lane.name}-{lane.calls}", batch_params(params))


class BatchLane:
    """One run's way into the batches. It stands in for the ``AsyncAnthropic`` client: ``messages.create`` queues the
    request in the ``Batcher`` and returns the message once its batch has ended."""

    def __init__(self, batcher: Batcher, name: str) -> None:
        self.batcher = batcher
        self.name = name  # custom_ids are ``<name>-<n>``, unique within the Batcher
        self.calls = 0
        self.messages = _LaneMessages(self)
        self.open = True

    def close(self) -> None:
        """The run will make no more calls, so the others no longer wait for it."""
        if self.open:
            self.open = False
            self.batcher._leave()


class BatchedWorkflowModel(AnthropicWorkflowModel):
    """``AnthropicWorkflowModel`` through a ``BatchLane``: the same requests, sent in batches, recorded at batch prices
    (so the run's Spend cap counts batch prices too). ``close`` it when the run ends."""

    def __init__(self, lane: BatchLane, model_id: str, *, max_tokens: int = 16000, effort: Effort | None = None) -> None:
        super().__init__(cast(anthropic.AsyncAnthropic, lane), model_id, max_tokens=max_tokens, effort=effort)
        self.lane = lane

    def _record(self, response: anthropic.types.Message) -> None:
        super()._record(response)
        self.calls[-1] = batch_priced(self.calls[-1])

    def close(self) -> None:
        self.lane.close()
