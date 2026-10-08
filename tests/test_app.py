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
