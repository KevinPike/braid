"""Validation gate M3.2 against a real Ollama: after compaction the model still recalls turn-5 facts.

Skipped unless HARNESS_LIVE=1 (needs Ollama and the daily driver pulled). The summarizer is
``HARNESS_SUMMARIZER`` or the daily driver.
"""

import os

import pytest
from strands.types.content import Message

from harness.agent import build_agent, stream_reply
from harness.context.compaction import CompactionPolicy, Compactor
from harness.context.ledger import Ledger
from harness.context.strategies import build_conversation_manager, make_summarizer
from harness.config import ContextConfig
from harness.context.trimlog import TrimLog
from harness.guard.ollama_client import OllamaClient

pytestmark = pytest.mark.skipif(os.environ.get("HARNESS_LIVE") != "1", reason="set HARNESS_LIVE=1 to run against Ollama")

FACTS = [
    ("My dog is called Biscuit.", "What is my dog called?", "biscuit"),
    ("The deploy password is tangerine-42.", "What is the deploy password?", "tangerine-42"),
    ("I was born in Lisbon.", "Where was I born?", "lisbon"),
    ("My sister Marta lives in Osaka.", "Which city does my sister live in?", "osaka"),
    ("The project deadline is 14 March.", "When is the project deadline?", "14 march"),
]


async def test_model_recalls_turn_five_facts_after_compaction() -> None:
    driver = os.environ.get("HARNESS_DRIVER", "gemma4:e4b")
    summarizer = os.environ.get("HARNESS_SUMMARIZER", driver)
    client = OllamaClient("http://localhost:11434")
    client.pin(driver, 8192)
    ledger = Ledger(client.pinned)
    trimlog = TrimLog()
    agent = build_agent(client, driver, "You answer in one short sentence.", ledger,
                        conversation_manager=build_conversation_manager(ContextConfig(strategy="summarize"), trimlog, ledger.estimator))
    msgs: list[Message] = agent.messages
    for i in range(1, 5):
        msgs += [{"role": "user", "content": [{"text": f"Small talk number {i}: how is the weather?"}]},
                 {"role": "assistant", "content": [{"text": "Mild and cloudy, thanks for asking."}]}]
    msgs += [{"role": "user", "content": [{"text": " ".join(f[0] for f in FACTS)}]},
             {"role": "assistant", "content": [{"text": "Noted."}]}]
    for i in range(20):
        msgs += [{"role": "user", "content": [{"text": f"Filler turn {i}: tell me something about topic {i}."}]},
                 {"role": "assistant", "content": [{"text": f"Topic {i} is a broad subject with many aspects worth exploring."}]}]
    helper = summarizer != driver
    compactor = Compactor(trimlog=trimlog, ledger=ledger, num_ctx=lambda: 8192,
                          summarize=make_summarizer(client, summarizer, 8192, ledger.estimator, unload_after=helper),
                          policy=CompactionPolicy(preserve_recent=6))
    entry = await compactor.compact(msgs, "live recall test")
    assert entry is not None and entry.tokens_after < entry.tokens_before

    correct = 0
    for _, question, answer in FACTS:
        snapshot = list(agent.messages)
        reply = "".join([d async for d in stream_reply(agent, question)])
        agent.messages[:] = snapshot  # questions must not teach each other
        correct += answer in reply.lower()
    await client.aclose()
    assert correct >= 4, f"recalled {correct}/5"
