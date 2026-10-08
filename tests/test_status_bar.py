from textual.app import App, ComposeResult

from harness.guard.watchdog import Alert, GuardState, Level
from harness.status import StatusBar, render_status

GIB = 1024**3


def healthy() -> GuardState:
    return GuardState(model="gemma4:e4b", gpu_fraction=1.0, size=6 * GIB, num_ctx=32768,
                      tokens_per_second=41.7, keep_alive_s=1725.0)


def test_render_shows_model_gpu_memory_tps_and_keepalive() -> None:
    text = render_status(healthy())
    for part in ("gemma4:e4b", "GPU 100%", "6.0 GiB", "42 tok/s", "keep-alive 29m", "ctx 32768"):
        assert part in text


def test_render_handles_no_model_loaded() -> None:
    assert "no model loaded" in render_status(GuardState())


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield StatusBar()


async def test_status_bar_colour_follows_level() -> None:
    app = Host()
    async with app.run_test() as pilot:
        bar = app.query_one(StatusBar)
        bar.set_state(healthy())
        assert bar.has_class("green")
        bar.set_state(GuardState(model="m", gpu_fraction=0.9, size=GIB, alerts=(Alert("spill", Level.RED, "x"),)))
        await pilot.pause()
        assert bar.has_class("red") and not bar.has_class("green")
        assert "GPU 90%" in str(bar.render())
        bar.set_state(GuardState(model="m", gpu_fraction=1.0, size=GIB, alerts=(Alert("memory_pressure", Level.YELLOW, "x"),)))
        assert bar.has_class("yellow")
