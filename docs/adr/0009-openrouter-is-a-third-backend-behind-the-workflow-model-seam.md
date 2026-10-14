# OpenRouter is a third backend, behind the WorkflowModel seam

Amends ADR 0007, which deferred OpenRouter "until cost or model choice calls for it". Model choice now does: the aim is to try many models in the **Workflows**, and to pair them (one writes the **Plan**, another codes), graded by the **Hidden tests**. Cost is tracked and capped, but it is not the reason. The design is in `specs/openrouter.md`.

## Decision

OpenRouter joins `anthropic` and `ollama` as a backend, chosen with `--backend openrouter` (the existing flag; not a new `--provider`, because OpenRouter already uses "provider" for the upstream host that serves a model) or by a `[workflow] backend`. It sits behind the existing `WorkflowModel` protocol, as `AnthropicWorkflowModel` over a client pointed at OpenRouter (see Transport below). The value is the shared framework around the model call:

- **Shared:** the `WorkflowModel` seam and its retry-once-then-`WorkflowModelError` contract; `ApiFailure` and the `api_error` status (a run stops resumable, F8); `ApiCall` usage and cost records; the **Spend cap**; Roles (F11); the Arbiter (F10); the bench ledger and frontier report (F2, F12); the run log and **Route records**.
- **Not shared:** the request and the response parsing. Each backend has its own failure patterns (OpenRouter: a `200` that carries a provider error, `402` when a key's credit limit is hit, `429` on free models, a routed model that ignores the schema). A failure on one is expected to strengthen the shared layer: `ApiFailure` and `API_FAULTS` move out of `anthropic_api.py` to a neutral module, and each backend maps its errors onto them (a later cleanup: today they still live in `anthropic_api.py`, and OpenRouter's errors arrive as the Anthropic SDK's own).

## Consequences

- **Data leaves the machine twice**: to OpenRouter and to the upstream provider. Free models often log prompts for training. `provider.data_collection = "deny"` is the default; free models need it set to `allow` and are for non-sensitive katas only.
- **A model without structured output is refused** (`structured_outputs` absent from its `supported_parameters`): `plan`, `build` and `bench` exit 2 before any call. Workflows need JSON, and the retry-once contract would hide how often a model fails to produce it.
- **Prices come from the response** (`usage.cost`), not `PRICING`, with `GET /api/v1/models` prices as the fallback. A `:free` model costs $0 toward the **Spend cap**.
- **`openrouter/auto` routes by OpenRouter's choice**, ranked by trailing-7-day spend share and bounded by `cost_tier` and `allowed_models`. It is not a graph edge, so it is outside ADR 0008's rule, but gets its spirit: every call records the served model (task type and tier are still to come), and a run on it is a dated measurement, not a repeatable row. Structured output is checked per reply, since the router is not documented to filter on it.
- **Transport.** OpenRouter's Anthropic-shaped endpoint (`https://openrouter.ai/api`) accepts the 1.x SDK's structured output, `thinking`, `cache_control`, `provider` routing and the `auto-router` plugin, so `AnthropicWorkflowModel` runs on it unchanged but for `extra_body`, `reported_cost` and `vet_served`. A `vendor/model` id with no dot in the vendor reads as OpenRouter, so an Ollama name such as `user/model` needs `--backend ollama`.
- **Credentials** come only from `OPENROUTER_API_KEY` in the environment. The key never appears in `braid.toml`, a run log, a summary or an error message (the `sk-or-` pattern is redacted), and the **Sandbox** env scrub (ADR 0005) removes it.
- `guard/` imports nothing for OpenRouter (ADR 0001). The Ollama path stays frozen (ADR 0007).
- The chat agent stays on Anthropic until a Workflow result says another model is worth moving it to.
