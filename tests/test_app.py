import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from textual.widgets import Input, Static

from harness.app import HarnessApp
from harness.config import HarnessConfig, load_config


async def fast_reply(prompt: str) -> AsyncIterator[str]:
    for tok in ["echo", ": ", prompt]:
        await asyncio.sleep(0)
        yield tok


async def slow_reply(prompt: str) -> AsyncIterator[str]:
    yield "start "
    await asyncio.sleep(60)
    yield "never"


def test_load_config(tmp_path: Path) -> None:
    p = tmp_path / "h.toml"
    p.write_text('[profile]\nmodel = "x"\nnum_ctx = 4096\n')
    cfg = load_config(p)
    assert (cfg.profile.model, cfg.profile.num_ctx) == ("x", 4096)
    assert load_config(tmp_path / "missing.toml") == HarnessConfig()


@pytest.mark.asyncio
async def test_ten_turn_chat_streams() -> None:
    app = HarnessApp(HarnessConfig(), replier=fast_reply)
    async with app.run_test() as pilot:
        for i in range(10):
            app.query_one(Input).value = f"msg{i}"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
        texts = [str(w.render()) for w in app.query(".assistant").results(Static)]
        assert texts == [f"echo: msg{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_ctrl_c_cancels_cleanly() -> None:
    app = HarnessApp(HarnessConfig(), replier=slow_reply)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "hi"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.generating
        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
        assert not app.generating
        # UI still accepts the next turn
        app.query_one(Input).value = "again"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.generating


@pytest.mark.asyncio
async def test_status_bar_follows_the_watchdog_and_goes_red_on_spill() -> None:
    from tests.guard.fakes import GIB, FakeBackend
    from harness.guard.watchdog import Watchdog
    from harness.status import StatusBar

    backend = FakeBackend(models={"m": (10 * GIB, 0.0)})
    backend.pins["m"] = 8192
    await backend.warm("m")
    wd = Watchdog(backend, pinned=backend.pins.get, interval=0.05, memory_level=lambda: 1)
    app = HarnessApp(HarnessConfig(), replier=fast_reply, watchdog=wd)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.query_one(StatusBar).has_class("green")
        backend.gpu_bytes = 4 * GIB  # a second model grabs the GPU
        await pilot.pause(0.3)
        assert app.query_one(StatusBar).has_class("red")
