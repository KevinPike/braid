"""Which backend a Workflow run uses, and its model (ADR 0006). The one place that decides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import anthropic

from harness.anthropic_api import make_client
from harness.config import API_PREFIX, HarnessConfig, RoleModel, WorkflowConfig, arbiter_role, role_model
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
    """The Arbiter (F10), always on the Anthropic API; None when neither ``[workflow.roles.arbiter] model`` nor
    ``[workflow] arbiter`` is set. Its Sonnet or Opus 5.5 requests opt in to server-side refusal fallbacks; no other
    model's do. ``client`` defaults to a new one."""
    role = arbiter_role(config.workflow)
    if role is None:
        return None
    if not role.model.startswith(API_PREFIX):
        raise ValueError(f"the Arbiter must be an Anthropic model id (claude-...), not {role.model!r}")
    return AnthropicWorkflowModel(
        client or make_client(config.anthropic), role.model, effort=role.effort,
        thinking_display=config.anthropic.thinking_display, fallbacks=role.model in FALLBACK_MODELS,
    )


class RoleModels(TypedDict, total=False):
    """A model per Role (F11), as ``build_execution_graph(**roles)`` takes them. A Role left out runs on the run's model
    (the Arbiter: there is none)."""

    tester: WorkflowModel
    implementer: WorkflowModel
    arbiter: WorkflowModel


def own_models(wf: WorkflowConfig, run_model: str) -> dict[Literal["tester", "implementer"], RoleModel]:
    """The tester and implementer whose ``[workflow.roles]`` model or effort differs from the run's; the others share
    the run's model object (so with no Roles configured nothing changes, not even the cache prefix)."""
    run = RoleModel(run_model, wf.effort)
    own: dict[Literal["tester", "implementer"], RoleModel] = {}
    for role in ("tester", "implementer"):
        if (rm := role_model(wf, role, run_model)) != run:
            if not rm.model.startswith(API_PREFIX):
                raise ValueError(f"[workflow.roles.{role}] needs an Anthropic model id (claude-...) for a model or effort "
                                 f"of its own, not {rm.model!r}")
            own[role] = rm
    return own


def make_roles(config: HarnessConfig, run_model: str, client: anthropic.AsyncAnthropic | None = None) -> RoleModels:
    """The Role models a run needs beyond its own (F11): the tester's and the implementer's when ``[workflow.roles]``
    gives them their own model or effort, always on the Anthropic API, and the Arbiter (``make_arbiter``). Raises
    ValueError for an Ollama or unpriced model, before a run. ``client`` defaults to one new client for all of them."""
    roles: RoleModels = {}
    for role, rm in own_models(config.workflow, run_model).items():
        client = client or make_client(config.anthropic)
        roles[role] = AnthropicWorkflowModel(client, rm.model, effort=rm.effort, thinking_display=config.anthropic.thinking_display)
    if (arbiter := make_arbiter(config, client)) is not None:
        roles["arbiter"] = arbiter
    return roles
