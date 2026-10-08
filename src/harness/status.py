"""Status bar: the Guard's output. Green on full GPU, yellow on pressure, red on spill or truncation."""

from __future__ import annotations

from textual.widgets import Static

from harness.guard.budget import GIB
from harness.guard.watchdog import GuardState, Level

_CLASSES = {Level.GREEN: "green", Level.YELLOW: "yellow", Level.RED: "red"}


def render_status(state: GuardState) -> str:
    if state.model is None:
        return "no model loaded"
    parts = [state.model]
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
    if state.alerts:
        parts.append(" | ".join(a.message for a in state.alerts))
    return "  ·  ".join(parts)


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar { height: 1; padding: 0 1; background: $panel; }
    StatusBar.green { background: $success 30%; }
    StatusBar.yellow { background: $warning 40%; }
    StatusBar.red { background: $error 60%; }
    """

    loading_model: str | None = None
    _state: GuardState = GuardState()

    def set_state(self, state: GuardState) -> None:
        self._state = state
        self._refresh_text()

    def show_loading(self, model: str | None) -> None:
        """While set, the bar says the model is loading instead of reporting a stale or empty ps."""
        self.loading_model = model
        self._refresh_text()

    def _refresh_text(self) -> None:
        if self.loading_model is not None:
            self.set_classes("yellow")
            self.update(f"loading {self.loading_model}…")
            return
        self.set_classes(_CLASSES[self._state.level])
        self.update(render_status(self._state))
