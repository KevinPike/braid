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


def test_loading_overrides_the_ps_state_until_cleared() -> None:
    bar = StatusBar()
    bar.set_state(healthy())
    bar.show_loading("gemma4:12b")
    assert bar.loading_model == "gemma4:12b"
    bar.show_loading(None)
    assert bar.loading_model is None


def test_render_shows_the_active_model_first_then_helpers_and_a_total_that_includes_both() -> None:
    from harness.guard.watchdog import ModelStatus

    state = GuardState(
        model="gemma4:12b", gpu_fraction=1.0, size=9 * GIB, num_ctx=16384, keep_alive_s=1200.0,
        models=(ModelStatus("gemma4:12b", 1.0, 9 * GIB, 16384, 1200.0), ModelStatus("tev1:0.8b", 1.0, GIB // 2, 2048, 600.0, True)),
    )
    text = render_status(state)
    assert text.index("gemma4:12b") < text.index("+ tev1:0.8b 0.5 GiB")
    assert "9.0 GiB" in text and "total 9.5 GiB" in text


def test_render_with_only_a_helper_loaded_says_so_and_still_lists_it() -> None:
    from harness.guard.watchdog import ModelStatus

    text = render_status(GuardState(models=(ModelStatus("tev1:0.8b", 1.0, GIB, 2048, 600.0, True),)))
    assert text.startswith("no model loaded") and "+ tev1:0.8b 1.0 GiB" in text  # no active model, but the helper is not hidden


async def test_the_bar_records_every_rendered_text_and_shows_the_run_clock_only_while_set() -> None:
    app = Host()
    async with app.run_test() as pilot:
        bar = app.query_one(StatusBar)
        seen: list[str] = []
        bar.on_render = seen.append
        bar.set_state(healthy())
        bar.set_run_elapsed(75)
        assert "workflow 1:15" in seen[-1]
        bar.set_run_elapsed(None)
        await pilot.pause()
        assert "workflow" not in seen[-1] and list(bar.history)[-len(seen):] == seen
