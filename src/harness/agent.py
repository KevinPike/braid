"""Strands Agent on the Ollama provider, exposed as a token stream."""

from __future__ import annotations

from collections.abc import AsyncIterator

from strands import Agent
from strands.models.ollama import OllamaModel

from harness.config import HarnessConfig


def build_agent(config: HarnessConfig) -> Agent:
    model = OllamaModel(
        config.ollama.host,
        model_id=config.profile.model,
        keep_alive=config.profile.keep_alive,
        options={"num_ctx": config.profile.num_ctx},
    )
    # callback_handler=None: the TUI renders the stream, nothing prints to stdout.
    return Agent(model=model, system_prompt=config.profile.system_prompt, callback_handler=None)


async def stream_reply(agent: Agent, prompt: str) -> AsyncIterator[str]:
    """Yield text deltas for one turn."""
    async for event in agent.stream_async(prompt):
        delta = event.get("data")
        if isinstance(delta, str) and delta:
            yield delta
