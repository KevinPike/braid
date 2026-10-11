"""Which backend a Workflow run uses, and its model (ADR 0006). The one place that decides."""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from harness.anthropic_api import make_client
from harness.config import API_PREFIX, HarnessConfig, arbiter_role
from harness.config import Backend as BackendName
from harness.guard.ollama_client import OllamaClient
from harness.workflow.model import FALLBACK_MODELS, AnthropicWorkflowModel, GuardedWorkflowModel, WorkflowModel


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
    built only for the anthropic backend, with the SDK's own credential resolution (a key is never passed in) and
    ``[anthropic] max_retries`` and ``timeout``."""
    wf = config.workflow
    if choice.backend == "anthropic":
        return AnthropicWorkflowModel(
            make_client(config.anthropic), choice.model_id, effort=wf.effort, thinking_display=config.anthropic.thinking_display,
        )
    if ollama is None:
        return None
    return GuardedWorkflowModel(ollama, choice.model_id, think=wf.think, num_predict=wf.num_predict)


def make_arbiter(config: HarnessConfig, client: anthropic.AsyncAnthropic | None = None) -> WorkflowModel | None:
    """The Arbiter (F10), always on the Anthropic API; None when ``[workflow] arbiter`` is unset. Its Sonnet or Opus 5.5
    requests opt in to server-side refusal fallbacks; no other model's do. ``client`` defaults to a new one."""
    role = arbiter_role(config.workflow)
    if role is None:
        return None
    if not role.model.startswith(API_PREFIX):
        raise ValueError(f"[workflow] arbiter must be an Anthropic model id (claude-...), not {role.model!r}")
    return AnthropicWorkflowModel(
        client or make_client(config.anthropic), role.model, effort=role.effort,
        thinking_display=config.anthropic.thinking_display, fallbacks=role.model in FALLBACK_MODELS,
    )
