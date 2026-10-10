"""Transcript inspector: each call's exact Ollama payload next to its estimated and measured token counts."""

from __future__ import annotations

import json
from collections.abc import Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from harness.session.store import CallEntry


def describe(call: CallEntry) -> str:
    """Header for the payload pane: the numbers the Ledger compared, then the request shape."""
    body = call.payload
    msgs = body.get("messages") or []
    tools = body.get("tools") or []
    est = "none (not ledgered)" if call.estimate is None else str(call.estimate)
    got = "pending" if call.prompt_eval_count is None else str(call.prompt_eval_count)
    err = "" if call.error is None else f" ({call.error:.1%} off)"
    evals = "" if call.eval_count is None else f", {call.eval_count} generated"
    return (f"call #{call.n} · {call.model} · num_ctx {call.num_ctx} · {len(msgs)} messages, {len(tools)} tools\n"
            f"estimate {est}; ollama prompt_eval_count {got}{err}{evals}")


class Inspector(ModalScreen[None]):
    DEFAULT_CSS = """
    Inspector { align: center middle; }
    #inspector { width: 95%; height: 90%; border: round $accent; background: $surface; padding: 0 1; }
    #inspector-calls { width: 44; border-right: solid $primary; }
    #inspector-body { padding: 0 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, calls: Sequence[CallEntry], selected: int | None = None) -> None:
        super().__init__()
        self.calls = list(calls)
        self._selected = selected

    def compose(self) -> ComposeResult:
        with Vertical(id="inspector"):
            yield Static(f"Transcript inspector: {len(self.calls)} calls (Esc closes)", markup=False)
            if not self.calls:
                yield Static("no calls recorded in this session yet", id="inspector-empty", markup=False)
                return
            with Horizontal():
                yield OptionList(*[Option(c.line(), id=str(c.n)) for c in self.calls], id="inspector-calls")
                with VerticalScroll(id="inspector-body"):
                    yield Static("", id="inspector-header", markup=False)
                    yield Static("", id="inspector-payload", markup=False)

    def on_mount(self) -> None:
        if not self.calls:
            return
        picks = [i for i, c in enumerate(self.calls) if c.n == self._selected]
        index = picks[0] if picks else len(self.calls) - 1
        options = self.query_one(OptionList)
        options.highlighted = index
        options.focus()
        self._show(self.calls[index])

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._show(self.calls[event.option_index])

    def _show(self, call: CallEntry) -> None:
        self.query_one("#inspector-header", Static).update(describe(call))
        self.query_one("#inspector-payload", Static).update(json.dumps(call.payload, indent=2, ensure_ascii=False))

    def action_close(self) -> None:
        self.dismiss(None)


_RUN_HEAD = ("ts", "behavior", "attempt", "round", "angle", "node_seconds", "model_seconds")
_RUN_TEXT = ("prompt", "prompts", "reply", "written", "refused", "exit_code", "timed_out", "output_tail", "answer", "reason", "verdict")


def describe_node(record: dict[str, object]) -> str:
    """A node's log record as the run inspector shows it: the numbers, then what was sent and what came back."""
    head = [f"{k} {record[k]}" for k in _RUN_HEAD if record.get(k) is not None]
    lines = [f"{record.get('node', '?')}  " + " · ".join(head)]
    shown = set(_RUN_HEAD) | set(_RUN_TEXT) | {"node"}
    for key in _RUN_TEXT:
        value = record.get(key)
        if value is None or value == [] or value == "":
            continue
        if key == "reply" or isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        lines.append(f"\n--- {key} ---\n{value}")
    rest = {k: v for k, v in record.items() if k not in shown and v not in (None, [], "")}
    if rest:
        lines.append("\n--- other ---\n" + json.dumps(rest, indent=2, ensure_ascii=False, default=str))
    return "\n".join(lines)


class RunInspector(ModalScreen[None]):
    """A Workflow run's JSONL log: one entry per node, with its prompt, reply, files, exit code and output tail."""

    DEFAULT_CSS = """
    RunInspector { align: center middle; }
    #run-inspector { width: 95%; height: 90%; border: round $accent; background: $surface; padding: 0 1; }
    #run-nodes { width: 44; border-right: solid $primary; }
    #run-body { padding: 0 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, run_id: str, records: Sequence[dict[str, object]]) -> None:
        super().__init__()
        self.run_id = run_id
        self.records = list(records)

    def compose(self) -> ComposeResult:
        with Vertical(id="run-inspector"):
            yield Static(f"Run {self.run_id}: {len(self.records)} node records (Esc closes)", markup=False)
            if not self.records:
                yield Static("the run log has no records yet", id="run-empty", markup=False)
                return
            with Horizontal():
                yield OptionList(*[Option(f"{i + 1:>3} {r.get('node', '?')}", id=str(i)) for i, r in enumerate(self.records)], id="run-nodes")
                with VerticalScroll(id="run-body"):
                    yield Static("", id="run-detail", markup=False)

    def on_mount(self) -> None:
        if not self.records:
            return
        options = self.query_one(OptionList)
        options.highlighted = 0
        options.focus()
        self._show(0)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._show(event.option_index)

    def _show(self, index: int) -> None:
        self.query_one("#run-detail", Static).update(describe_node(self.records[index]))

    def action_close(self) -> None:
        self.dismiss(None)
