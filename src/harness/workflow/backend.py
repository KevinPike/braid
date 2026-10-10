"""Which backend a Workflow run uses, and its model (ADR 0006). The one place that decides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anthropic

from harness.config import HarnessConfig
from harness.guard.ollama_client import OllamaClient
from harness.workflow.model import AnthropicWorkflowModel, GuardedWorkflowModel, WorkflowModel

BackendName = Literal["ollama", "anthropic"]
API_PREFIX = "claude-"


@dataclass(frozen=True)
class Choice:
    backend: BackendName
    model_id: str


def choose(config: HarnessConfig, *, model: str | None = None, backend: BackendName | None = None) -> Choice:
    """An explicit ``backend`` wins, then the model id (``claude-`` is anthropic, anything else ollama), then
    ``[workflow] backend``. Without a model id, each backend reads its own: ``anthropic_model``, or ``model`` then the
    Profile's model."""
    wf = config.workflow
    by_model: BackendName | None = None if not model else "anthropic" if model.startswith(API_PREFIX) else "ollama"
    picked: BackendName = backend or by_model or wf.backend
    if model:
        return Choice(picked, model)
    return Choice(picked, wf.anthropic_model if picked == "anthropic" else wf.model or config.profile.model)


def make_model(config: HarnessConfig, choice: Choice, ollama: OllamaClient | None) -> WorkflowModel | None:
    """The WorkflowModel for ``choice``; None when it needs an Ollama client and there is none. The Anthropic client is
    built only for the anthropic backend, with the SDK's own credential resolution (a key is never passed in)."""
    wf = config.workflow
    if choice.backend == "anthropic":
        return AnthropicWorkflowModel(anthropic.AsyncAnthropic(), choice.model_id, effort=wf.effort)
    if ollama is None:
        return None
    return GuardedWorkflowModel(ollama, choice.model_id, think=wf.think, num_predict=wf.num_predict)
