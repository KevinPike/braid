"""Chat widgets: a markdown reply that re-renders from its full text, the context panel and the alert banner."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Markdown, Static

from harness.approval import ApprovalRequest
from harness.choice import OTHER, Answer, ChoiceRequest, ChoiceResult, ChoiceStatus, clip_preview, redact, redact_request
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
            f"estimate {record.estimate} · {'api' if record.model.startswith('claude-') else 'ollama'} {record.prompt_eval_count}"
            f" · out {record.eval_count}"
            + (f" · off {err:.1%}" if err is not None else "")
        )
    if record.strands_estimate is not None:
        nums += f" · strands {record.strands_estimate}"
    return f"{render_gauge(cats, record.num_ctx)} {used}/{record.num_ctx} ({pct})\n{legend}\n{nums}"


def render_loaded(record: CallRecord | None, state: GuardState | None) -> str:
    """Which model is loaded right now, and (when the last call was on another one) whose numbers the gauge shows."""
    if state is None or state.model is None:
        return ""
    text = f"loaded: {state.model}" + (f" at num_ctx {state.num_ctx}" if state.num_ctx else "")
    if record is not None and record.model != state.model:
        text += f" (gauge: last call on {record.model})"
    return text


class ContextPanel(Static):
    DEFAULT_CSS = "ContextPanel { height: 3; padding: 0 1; background: $panel; }"

    _record: CallRecord | None = None
    _state: GuardState | None = None

    def set_record(self, record: CallRecord | None) -> None:
        self._record = record
        self._redraw()

    def set_state(self, state: GuardState) -> None:
        """The Watchdog's latest poll: the panel names the model that is loaded, so a swap shows within one poll."""
        if state != self._state:
            self._state = state
            self._redraw()

    def _redraw(self) -> None:
        text = render_context(self._record)
        loaded = render_loaded(self._record, self._state)
        self.update(f"{text}\n{loaded}" if loaded and self._record is None else f"{text} · {loaded}" if loaded else text)


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


class OtherInput(Input):
    """The free-text field of a ``ChoicePrompt`` Other option."""


class NoteInput(Input):
    """The optional note field of a ``ChoicePrompt``."""


class ChoicePrompt(Vertical, can_focus=True):
    """A Choice asked in the chat log (ADR 0010): one to four questions, an option list with an optional preview, a free
    text Other and a note. Mounted by ``HarnessApp.ask_choice`` and replaced by a one-line record once it is answered.

    It shows the redacted request but answers with the asker's real labels. Esc declines; the app resolves it as
    cancelled on Ctrl-C. An Input inside it must not swallow Esc or Ctrl-C, so both are bound here and on the app."""

    DEFAULT_CSS = """
    ChoicePrompt { height: auto; padding: 0 1; margin: 1 0; border: round $warning; }
    ChoicePrompt Static { height: auto; }
    ChoicePrompt #cols { height: auto; }
    ChoicePrompt #options { width: 1fr; }
    ChoicePrompt #preview { width: 1fr; border-left: tall $primary; padding-left: 1; display: none; }
    ChoicePrompt.narrow #cols { layout: vertical; }
    ChoicePrompt.narrow #preview { border-left: none; border-top: tall $primary; padding-left: 0; }
    ChoicePrompt Input { height: 3; display: none; }
    ChoicePrompt Input.open { display: block; }
    ChoicePrompt #hint { color: $text-muted; }
    """
    BINDINGS = [
        *[Binding(str(n), f"pick({n})", show=False) for n in range(1, 6)],
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
        Binding("space", "tick", show=False),
        Binding("enter", "next", show=False),
        Binding("tab", "go(1)", show=False),
        Binding("shift+tab", "go(-1)", show=False),
        Binding("n", "note", show=False),
        Binding("escape", "decline", show=False),
    ]
    HINT = "1-4 pick · ↑↓ move · space toggle (multi) · enter next · tab/⇧tab question · n note · esc decline"

    def __init__(self, request: ChoiceRequest, model: str = "", source: str = "chat") -> None:
        super().__init__(classes="decision pending")
        self.request = request
        self.source = source
        self._shown = redact_request(request)  # what the human reads; answers carry the asker's own labels
        self.border_title = f"Choice from {model}" if model else "Choice"
        n = len(request.questions)
        self._q = 0
        self._cursor = [0] * n
        self._picked: list[set[int]] = [set() for _ in range(n)]  # option indexes; len(options) is Other
        self._other: list[str] = [""] * n
        self._notes: list[str] = [""] * n
        self._done = [False] * n
        self._message = ""
        self._future: asyncio.Future[ChoiceResult] | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="head", markup=False)
        yield Static("", id="ask", markup=False)
        with Horizontal(id="cols"):
            yield Static("", id="options", markup=False)
            yield Static("", id="preview", markup=False)
        yield OtherInput(placeholder="your own answer, then Enter", id="other")
        yield NoteInput(placeholder="note (optional), then Enter", id="note")
        yield Static("", id="hint", markup=False)

    def on_mount(self) -> None:
        self._redraw()

    def on_resize(self) -> None:
        self.set_class(self.size.width < 100, "narrow")

    @property
    def is_open(self) -> bool:
        return self._future is not None and not self._future.done()

    async def answer(self) -> ChoiceResult:
        """Wait for the Choice to be answered, declined or cancelled. Cancelling the waiting task (Ctrl-C) answers nothing."""
        self._future = asyncio.get_running_loop().create_future()
        return await self._future

    def resolve(self, status: ChoiceStatus, reason: str = "") -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(ChoiceResult(status=status, reason=reason))

    def record(self, result: ChoiceResult, seconds: float) -> str:
        """The one-line record the app leaves in the chat log once this is over."""
        if result.status != "answered":
            return f"✗ choice {result.status.replace('_', ' ')}"
        parts = " · ".join(f"{q.header}? {redact(result.answers[q.question].text())}" for q in self._shown.questions
                           if q.question in result.answers)
        return f"✓ {parts} ({seconds:.1f} s)"

    # --- state ---------------------------------------------------------------------------------------------------

    def _n_options(self) -> int:
        return len(self.request.questions[self._q].options)

    def _redraw(self) -> None:
        q, shown = self._q, self._shown.questions[self._q]
        total = len(self._shown.questions)
        chips = " ".join((f"[{s.header}{' ✓' if self._done[i] else ''}]" if i == q else f" {s.header}{' ✓' if self._done[i] else ''} ")
                         for i, s in enumerate(self._shown.questions))
        self.query_one("#head", Static).update(f"{chips}    question {q + 1}/{total}")
        self.query_one("#ask", Static).update(shown.question)
        rows = []
        for i, o in enumerate(shown.options):
            mark = ("[x] " if i in self._picked[q] else "[ ] ") if shown.multi_select else ""
            rows.append(f"{'❯' if i == self._cursor[q] else ' '} {i + 1} {mark}{o.label}" + (f"  {o.description}" if o.description else ""))
        other = len(shown.options)
        typed = f"Other: {self._other[q]}" if self._other[q] else f"{OTHER}…  type your own answer"
        mark = ("[x] " if other in self._picked[q] else "[ ] ") if shown.multi_select else ""
        rows.append(f"{'❯' if other == self._cursor[q] else ' '} {other + 1} {mark}{typed}")
        self.query_one("#options", Static).update("\n".join(rows))
        cursor = self._cursor[q]
        preview = shown.options[cursor].preview if cursor < other else None
        pane = self.query_one("#preview", Static)
        pane.display = bool(preview)
        pane.update(clip_preview(preview) if preview else "")
        self.query_one("#hint", Static).update(self._message or self.HINT)
        note = self.query_one("#note", NoteInput)
        if note.value != self._notes[q] and not note.has_focus:
            note.value = self._notes[q]

    def _say(self, message: str) -> None:
        self._message = message
        self._redraw()

    def _go(self, q: int) -> None:
        self._close_other()
        self._q = q
        self._message = ""
        note = self.query_one("#note", NoteInput)
        note.value = self._notes[q]
        self._redraw()

    def _open_other(self) -> None:
        box = self.query_one("#other", OtherInput)
        box.value = self._other[self._q]
        box.add_class("open")
        box.focus()

    def _close_other(self) -> None:
        self.query_one("#other", OtherInput).remove_class("open")
        if self.app.focused is not self:
            self.focus()

    # --- keys ----------------------------------------------------------------------------------------------------

    def action_pick(self, number: int) -> None:
        q, other = self._q, self._n_options()
        if not 1 <= number <= other + 1:
            return
        self._cursor[q] = number - 1
        self._message = ""
        if number - 1 == other:
            self._open_other()
            self._redraw()
            return
        if self.request.questions[q].multi_select:
            self._picked[q] ^= {number - 1}
        else:
            self._picked[q] = {number - 1}
        self._redraw()

    def action_move(self, step: int) -> None:
        q = self._q
        self._cursor[q] = (self._cursor[q] + step) % (self._n_options() + 1)
        self._message = ""
        self._redraw()

    def action_tick(self) -> None:
        if self.request.questions[self._q].multi_select:
            self.action_pick(self._cursor[self._q] + 1)

    def action_go(self, step: int) -> None:
        self._go((self._q + step) % len(self.request.questions))

    def action_note(self) -> None:
        self.query_one("#note", NoteInput).focus()

    def action_decline(self) -> None:
        self.resolve("declined", "the user declined to answer")

    def action_next(self) -> None:
        """Accept the current question (the highlighted option, or the ticked ones) and move on, or submit the last."""
        q, spec = self._q, self.request.questions[self._q]
        other = len(spec.options)
        if not spec.multi_select:
            self._picked[q] = {self._cursor[q]}
        picked = self._picked[q]
        if not picked:
            self._say("pick at least one option (space ticks, or press its number)")
            return
        if other in picked and not self._other[q]:
            self._open_other()
            return
        self._done[q] = True
        self._close_other()
        waiting = [i for i in range(len(self._done)) if not self._done[i]]
        if waiting:
            self._go(next((i for i in waiting if i > q), waiting[0]))
        else:
            self._submit()

    def _submit(self) -> None:
        answers: dict[str, Answer] = {}
        for i, spec in enumerate(self.request.questions):
            labels = tuple(o.label for j, o in enumerate(spec.options) if j in self._picked[i])
            has_other = len(spec.options) in self._picked[i]
            answers[spec.question] = Answer(
                picked=(*labels, OTHER) if has_other else labels, other=self._other[i] if has_other else None,
                note=self._notes[i])
        if self._future is not None and not self._future.done():
            self._future.set_result(ChoiceResult(status="answered", answers=answers))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()  # the app's own handler is for the prompt box
        q = self._q
        if isinstance(event.input, NoteInput):
            self._notes[q] = event.value.strip()
            self.focus()
        elif isinstance(event.input, OtherInput):
            text = event.value.strip()
            if not text:
                self._say("Other needs some text (Esc declines the whole Choice)")
                return
            self._other[q] = text
            self._picked[q] = self._picked[q] | {len(self.request.questions[q].options)} if self.request.questions[q].multi_select \
                else {len(self.request.questions[q].options)}
            self._cursor[q] = len(self.request.questions[q].options)
            self.action_next()

    def on_input_changed(self, event: Input.Changed) -> None:
        if isinstance(event.input, NoteInput) and event.input.has_focus:  # not the redraw that fills it in
            self._notes[self._q] = event.value.strip()
        event.stop()
