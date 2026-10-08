"""Validation gates M4.1 and M4.2 against a real Ollama (HARNESS_LIVE=1).

Gate 1: on 20 prompts the daily driver picks the right tool (or none) in at least 16.
Gate 2: the tev1 router agrees with the hand labels on at least 16, and every decision is logged.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from strands import Agent

from harness.agent import GuardedOllamaModel
from harness.decide.log import DecisionLog
from harness.decide.questions import route
from harness.decide.tev import TevClient
from harness.guard.ollama_client import OllamaClient
from harness.tools.builtin import build_tools

pytestmark = pytest.mark.skipif(os.environ.get("HARNESS_LIVE") != "1", reason="set HARNESS_LIVE=1 to run against Ollama")

PROMPTS: list[dict[str, Any]] = json.loads((Path(__file__).parent / "data" / "prompts.json").read_text())
HOST = "http://localhost:11434"


async def test_driver_picks_the_right_tool() -> None:
    driver = os.environ.get("HARNESS_DRIVER", "gemma4:e4b")
    client = OllamaClient(HOST)
    client.pin(driver, 8192)
    tools: list[Any] = list(build_tools(Path("."), Path("notes")))
    correct, misses = 0, []
    for case in PROMPTS:
        agent = Agent(model=GuardedOllamaModel(client, driver), system_prompt="You are a concise assistant. Use a tool only when the request needs one.",
                      callback_handler=None, tools=tools)
        await agent.invoke_async(case["prompt"])
        used = [b["toolUse"]["name"] for m in agent.messages for b in m["content"] if "toolUse" in b]
        first = used[0] if used else None
        if first == case["tool"]:
            correct += 1
        else:
            misses.append((case["prompt"], case["tool"], first))
    await client.aclose()
    print(f"\ntool selection: {correct}/20")
    assert correct >= 16, f"{correct}/20 correct; misses: {misses}"


async def test_tev1_router_agrees_with_hand_labels_and_logs_every_decision() -> None:
    log = DecisionLog()
    tev = TevClient(HOST, os.environ.get("HARNESS_TEV", "tev1:0.8b"), log=log)
    correct, misses = 0, []
    for case in PROMPTS:
        label = "chat" if case["tool"] is None else "tool"
        choice, _ = await route(tev, case["prompt"])
        if choice == label:
            correct += 1
        else:
            misses.append((case["prompt"], label, choice))
    await tev.aclose()
    entries = log.entries()
    assert len(entries) == 20 and all(e.answers["route"]["probabilities"] for e in entries)
    print(f"\nrouter agreement: {correct}/20")
    assert correct >= 16, f"{correct}/20 agree; misses: {misses}"
