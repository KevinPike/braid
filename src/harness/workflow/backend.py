"""Which backend a Workflow run uses, and its model (ADR 0006). The one place that decides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import anthropic

from harness.anthropic_api import ApiFailure, make_client, make_openrouter_client, openrouter_body
from harness.config import API_PREFIX, OPENROUTER_AUTO, Effort, HarnessConfig, RoleModel, WorkflowConfig, arbiter_role, is_openrouter_id, role_model
from harness.config import Backend as BackendName
from harness.guard.ollama_client import OllamaClient
from harness.openrouter import Catalog, KeyStatus, key_status, load_catalog, refuse_unstructured, vet_allow_list
from harness.workflow.model import FALLBACK_MODELS, AnthropicWorkflowModel, GuardedWorkflowModel, WorkflowModel


@dataclass(frozen=True)
class Choice:
    backend: BackendName
    model_id: str


def choose(config: HarnessConfig, *, model: str | None = None, backend: BackendName | None = None) -> Choice:
    """An explicit ``backend`` wins, then the model id (``claude-`` is anthropic, ``vendor/model`` openrouter, anything
    else ollama), then ``[workflow] backend``. Without a model id, each backend reads its own: ``anthropic_model``,
    ``openrouter_model``, or ``model`` then the Profile's model. OpenRouter has no default model: ValueError."""
    wf = config.workflow
    by_model: BackendName | None = (
        None if not model else "anthropic" if model.startswith(API_PREFIX) else "openrouter" if is_openrouter_id(model) else "ollama"
    )
    picked: BackendName = backend or by_model or wf.backend
    if model:
        return Choice(picked, model)
    if picked == "openrouter":
        if not wf.openrouter_model:
            raise ValueError("the openrouter backend needs --model vendor/model (or [workflow] openrouter_model)")
        return Choice(picked, wf.openrouter_model)
    return Choice(picked, wf.anthropic_model if picked == "anthropic" else wf.model or config.profile.model)


def openrouter_model(
    config: HarnessConfig, model_id: str, *, effort: Effort | None, catalog: Catalog | None, client: anthropic.AsyncAnthropic | None = None,
) -> AnthropicWorkflowModel:
    """A WorkflowModel on OpenRouter (ADR 0009): ``AnthropicWorkflowModel`` over the OpenRouter client, priced by the
    response. With a ``catalog``, any model that answers without structured output fails the call (``openrouter/auto``
    chooses its own); a served id the catalog does not know passes and is recorded."""
    vet = None if catalog is None else (lambda served: _vet_served(catalog, served))
    return AnthropicWorkflowModel(
        client or make_openrouter_client(config.openrouter), model_id, effort=effort, max_tokens=config.openrouter.max_tokens,
        thinking_display=config.anthropic.thinking_display, thinking_sample=config.anthropic.thinking_sample, slow_after_s=config.workflow.slow_call_s or None, extra_body=openrouter_body(config.openrouter, model_id),
        reported_cost=True, vet_served=vet,
    )


def _vet_served(catalog: Catalog, served: str) -> None:
    known = catalog.get(served)
    if known is not None and not known.structured_outputs:
        raise ApiFailure("UnstructuredModel", f"{served} answered but does not support structured outputs")


def openrouter_ids(config: HarnessConfig, run_model: str | None) -> list[str]:
    """Every OpenRouter model id a run would call: the run's, and the Roles' and the Arbiter's that are OpenRouter ids."""
    wf = config.workflow
    ids = [run_model] if run_model and is_openrouter_id(run_model) else []
    ids += [r.model for r in (wf.roles.tester, wf.roles.implementer) if r.model and is_openrouter_id(r.model)]
    if (role := arbiter_role(wf)) is not None and is_openrouter_id(role.model):
        ids.append(role.model)
    return list(dict.fromkeys(ids))


async def prepare_openrouter(config: HarnessConfig, model_ids: list[str]) -> tuple[Catalog, KeyStatus]:
    """Everything OpenRouter needs checked before the first call (ADR 0009): the key, every model in the catalog with
    structured output (``ModelRefused`` otherwise), and for ``openrouter/auto`` an allow-list of such models. Returns the
    catalog and the key's credit. ValueError names what is missing."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set (put it in .env and run with `uv run --env-file .env`)")
    catalog = await load_catalog(api_key=key, cache_path=config.paths.openrouter_models_file)
    refuse_unstructured(catalog, model_ids)
    if OPENROUTER_AUTO in model_ids:
        auto = config.openrouter.auto
        vet_allow_list(catalog, auto.allowed_models, auto.excluded_models)
    return catalog, await key_status(api_key=key)


def make_model(
    config: HarnessConfig, choice: Choice, ollama: OllamaClient | None, catalog: Catalog | None = None,
) -> WorkflowModel | None:
    """The WorkflowModel for ``choice``; None when it needs an Ollama client and there is none. The Anthropic client is
    built only for the anthropic backend, with the SDK's own credential resolution (a key is never passed in) and
    ``[anthropic] max_retries`` and ``timeout``. ``catalog`` is the OpenRouter catalog that vets each served model."""
    wf = config.workflow
    if choice.backend == "openrouter":
        return openrouter_model(config, choice.model_id, effort=wf.effort, catalog=catalog)
    if choice.backend == "anthropic":
        return AnthropicWorkflowModel(
            make_client(config.anthropic), choice.model_id, effort=wf.effort, thinking_display=config.anthropic.thinking_display, thinking_sample=config.anthropic.thinking_sample, slow_after_s=config.workflow.slow_call_s or None,
        )
    if ollama is None:
        return None
    return GuardedWorkflowModel(ollama, choice.model_id, think=wf.think, num_predict=wf.num_predict)


def make_arbiter(
    config: HarnessConfig, client: anthropic.AsyncAnthropic | None = None, catalog: Catalog | None = None,
) -> WorkflowModel | None:
    """The Arbiter (F10), always on the Anthropic API; None when neither ``[workflow.roles.arbiter] model`` nor
    ``[workflow] arbiter`` is set. Its Sonnet or Opus 5.5 requests opt in to server-side refusal fallbacks; no other
    model's do. ``client`` defaults to a new one."""
    role = arbiter_role(config.workflow)
    if role is None:
        return None
    if is_openrouter_id(role.model):  # ADR 0009: an Arbiter on OpenRouter, priced by the response
        return openrouter_model(config, role.model, effort=role.effort, catalog=catalog)
    if not role.model.startswith(API_PREFIX):
        raise ValueError(f"the Arbiter must be an Anthropic (claude-...) or OpenRouter (vendor/model) id, not {role.model!r}")
    return AnthropicWorkflowModel(
        client or make_client(config.anthropic), role.model, effort=role.effort,
        thinking_display=config.anthropic.thinking_display, thinking_sample=config.anthropic.thinking_sample, slow_after_s=config.workflow.slow_call_s or None, fallbacks=role.model in FALLBACK_MODELS,
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
            if not (rm.model.startswith(API_PREFIX) or is_openrouter_id(rm.model)):
                raise ValueError(f"[workflow.roles.{role}] needs an Anthropic (claude-...) or OpenRouter (vendor/model) id for a "
                                 f"model or effort of its own, not {rm.model!r}")
            own[role] = rm
    return own


def make_roles(
    config: HarnessConfig, run_model: str, client: anthropic.AsyncAnthropic | None = None, catalog: Catalog | None = None,
) -> RoleModels:
    """The Role models a run needs beyond its own (F11): the tester's and the implementer's when ``[workflow.roles]``
    gives them their own model or effort, always on the Anthropic API, and the Arbiter (``make_arbiter``). Raises
    ValueError for an Ollama or unpriced model, before a run. ``client`` defaults to one new client for all of them."""
    roles: RoleModels = {}
    for role, rm in own_models(config.workflow, run_model).items():
        if is_openrouter_id(rm.model):
            roles[role] = openrouter_model(config, rm.model, effort=rm.effort, catalog=catalog)
            continue
        client = client or make_client(config.anthropic)
        roles[role] = AnthropicWorkflowModel(client, rm.model, effort=rm.effort, thinking_display=config.anthropic.thinking_display, thinking_sample=config.anthropic.thinking_sample, slow_after_s=config.workflow.slow_call_s or None)
    if (arbiter := make_arbiter(config, client, catalog)) is not None:
        roles["arbiter"] = arbiter
    return roles
