"""Chat widgets: a markdown reply that re-renders from its full text, the context panel and the alert banner."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Markdown, Static

from harness.approval import ApprovalRequest

from harness.context.accounting import Category
from harness.context.ledger import CallRecord
from harness.guard.watchdog import GuardState, Level

_THROTTLE_S = 0.08


class Reply(Markdown):
    """Assistant reply. Always rendered from the whole accumulated text, so any chunking of a
    stream gives the same final result as rendering it at once (an unclosed fence is just code to the end)."""

    DEFAULT_CSS = """
    Reply { margin: 0 0 1 0; padding: 0; }
    Reply.note { color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self.text = ""
        self._dirty = False

    def feed(self, delta: str) -> None:
        self.text += delta
        if not self._dirty:
            self._dirty = True
            self.set_timer(_THROTTLE_S, self._flush)

    async def _flush(self) -> None:
        self._dirty = False
        await self.update(self.text)

    async def finish(self, text: str | None = None) -> None:
        """Render the final text now (cancels nothing; a pending throttled flush becomes a no-op rewrite)."""
        if text is not None:
            self.text = text
        self._dirty = False
        await self.update(self.text)


_SEGMENT_STYLES = {
    Category.SYSTEM: "blue",
    Category.TOOLS: "magenta",
    Category.HISTORY: "cyan",
    Category.TOOL_RESULTS: "yellow",
    Category.TURN: "green",
}
GAUGE_WIDTH = 60


def render_gauge(categories: dict[Category, int] | None, num_ctx: int, width: int = GAUGE_WIDTH) -> str:
    """Stacked bar of category shares of ``num_ctx``, as Rich markup; free space is dim dots."""
    if not num_ctx or not categories:
        return "[dim]" + "·" * width + "[/]"
    out, used = "", 0
    for cat in Category:
        cells = round(categories.get(cat, 0) / num_ctx * width)
        cells = min(cells, width - used)
        if cells > 0:
            out += f"[{_SEGMENT_STYLES[cat]}]" + "█" * cells + "[/]"
            used += cells
    return out + "[dim]" + "·" * (width - used) + "[/]"


def render_context(record: CallRecord | None) -> str:
    if record is None:
        return "context: no calls yet"
    cats = dict(record.scaled_categories())
    legend = "  ".join(f"[{_SEGMENT_STYLES[c]}]■[/] {c.value} {cats[c]}" for c in Category)
    used = record.prompt_eval_count if record.prompt_eval_count is not None else record.estimate
    pct = f"{used / record.num_ctx:.0%}" if record.num_ctx else "?"
    if record.prompt_eval_count is None:
        nums = f"estimate {record.estimate} (in flight)"
    else:
        err = record.error
        nums = (
            f"estimate {record.estimate} · ollama {record.prompt_eval_count} · out {record.eval_count}"
            + (f" · off {err:.1%}" if err is not None else "")
        )
    if record.strands_estimate is not None:
        nums += f" · strands {record.strands_estimate}"
    return f"{render_gauge(cats, record.num_ctx)} {used}/{record.num_ctx} ({pct})\n{legend}\n{nums}"


class ContextPanel(Static):
    DEFAULT_CSS = "ContextPanel { height: 3; padding: 0 1; background: $panel; }"

    def set_record(self, record: CallRecord | None) -> None:
        self.update(render_context(record))


class AlertBanner(Static):
    """Red banner; visible only while a red Alert is active."""

    DEFAULT_CSS = """
    AlertBanner { display: none; height: auto; padding: 0 1; background: $error; color: $text; }
    AlertBanner.active { display: block; }
    """

    def set_state(self, state: GuardState) -> None:
        reds = [a.message for a in state.alerts if a.level == Level.RED]
        self.set_class(bool(reds), "active")
        self.update("\n".join(reds))


class ApprovalPrompt(Static, can_focus=True):
    """Asks the user to approve one action; mounted into the chat log by ``HarnessApp.ask_approval``, not part of
    the layout. Takes focus so y and n reach it; the app replaces it with a one-line record once it is answered."""

    DEFAULT_CSS = """
    ApprovalPrompt { height: auto; padding: 0 1; margin: 1 0; border: round $warning; }
    """
    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__(
            f"Approve {request.tool}?   y approve · n or Esc deny\n{request.describe()}\n{request.advice.line()}",
            markup=False, classes="decision pending",
        )
        self.request = request
        self._future: asyncio.Future[bool] | None = None

    @property
    def is_open(self) -> bool:
        return self._future is not None and not self._future.done()

    async def answer(self) -> bool:
        """Wait for y or n. Cancelling the waiting task (Ctrl-C) leaves the action unapproved."""
        self._future = asyncio.get_running_loop().create_future()
        return await self._future

    def decide(self, approved: bool) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(approved)

    def action_approve(self) -> None:
        self.decide(True)

    def action_deny(self) -> None:
        self.decide(False)


class AnswerInput(Input):
    """The text field of an ``AnswerPrompt``."""


class AnswerPrompt(Vertical):
    """A Workflow interrupt (Stuck, clarify, Sign-off, review) asked in the chat log: the question and a field for
    the answer. The app replaces it with a one-line record once it is answered."""

    DEFAULT_CSS = """
    AnswerPrompt { height: auto; padding: 0 1; margin: 1 0; border: round $warning; }
    AnswerPrompt Static { height: auto; }
    AnswerPrompt Input { margin-top: 1; }
    """

    def __init__(self, body: str, hint: str) -> None:
        super().__init__(classes="decision pending")
        self.body = body
        self.hint = hint
        self._future: asyncio.Future[str] | None = None

    def compose(self) -> ComposeResult:
        yield Static(self.body, markup=False)
        yield AnswerInput(placeholder=self.hint)

    def focus_field(self) -> None:
        self.query_one(AnswerInput).focus()

    async def answer(self) -> str:
        self._future = asyncio.get_running_loop().create_future()
        return await self._future

    def decide(self, text: str) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()  # the app's own handler is for the prompt box, not for answers
        if event.value.strip():
            self.decide(event.value.strip())
