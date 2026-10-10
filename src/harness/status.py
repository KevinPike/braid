"""Status bar: the Guard's output. Green on full GPU, yellow on pressure, red on spill or truncation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from textual.widgets import Static

from harness.guard.budget import GIB
from harness.guard.watchdog import GuardState, Level

_CLASSES = {Level.GREEN: "green", Level.YELLOW: "yellow", Level.RED: "red"}


def render_status(state: GuardState) -> str:
    """The active model first, then its numbers, then every other loaded model (helpers) and the memory total."""
    parts = [state.model or "no model loaded"]
    if state.gpu_fraction is not None:
        parts.append(f"GPU {state.gpu_fraction:.0%}")
    if state.size is not None:
        parts.append(f"{state.size / GIB:.1f} GiB")
    if state.num_ctx:
        parts.append(f"ctx {state.num_ctx}")
    if state.tokens_per_second is not None:
        parts.append(f"{state.tokens_per_second:.0f} tok/s")
    if state.keep_alive_s is not None:
        parts.append(f"keep-alive {max(0, round(state.keep_alive_s / 60))}m")
    for helper in state.helpers:
        parts.append(f"+ {helper.name} {helper.size / GIB:.1f} GiB (GPU {helper.gpu_fraction:.0%})")
    if state.helpers and state.total_size is not None:
        parts.append(f"total {state.total_size / GIB:.1f} GiB")
    if state.alerts:
        parts.append(" | ".join(a.message for a in state.alerts))
    return "  ·  ".join(parts)


def format_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar { height: 1; padding: 0 1; background: $panel; }
    StatusBar.green { background: $success 30%; }
    StatusBar.yellow { background: $warning 40%; }
    StatusBar.red { background: $error 60%; }
    """

    loading_model: str | None = None
    _state: GuardState = GuardState()
    _elapsed: float | None = None
    on_render: Callable[[str], None] | None = None  # sees every text the bar renders (the debug log)

    def __init__(self, content: str = "", *, id: str | None = None) -> None:
        super().__init__(content, id=id)
        self.history: deque[str] = deque(maxlen=500)  # what was rendered, oldest first

    def set_state(self, state: GuardState) -> None:
        self._state = state
        self._refresh_text()

    def show_loading(self, model: str | None) -> None:
        """While set, the bar says the model is loading instead of reporting a stale or empty ps."""
        self.loading_model = model
        self._refresh_text()

    def set_run_elapsed(self, seconds: float | None) -> None:
        """Elapsed time of the active Workflow run, or None when no run is active."""
        self._elapsed = seconds
        self._refresh_text()

    def _refresh_text(self) -> None:
        if self.loading_model is not None:
            self.set_classes("yellow")
            text = f"loading {self.loading_model}…"
        else:
            self.set_classes(_CLASSES[self._state.level])
            text = render_status(self._state)
        if self._elapsed is not None:
            text += f"  ·  workflow {format_elapsed(self._elapsed)}"
        self.history.append(text)
        if self.on_render is not None:
            self.on_render(text)
        self.update(text)
