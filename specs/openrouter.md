# OpenRouter: a third backend, for cheap and new-to-me models

Drafted 2026-10-13. Terms are from `CONTEXT.md`. ADR 0007 deferred OpenRouter ("revisit when cost or model choice calls for it"); this is that revisit. Decided in **ADR 0009**; status and probe results are under "Status" below.

**Goal.** Try many models in braid's Workflows, and pair them: one model writes the **Plan**, another does the coding. Trying a new model is one flag, and each pairing is graded by the **Hidden tests** and compared on a table. Cost is tracked and capped but is not the aim (revised 2026-10-13: the user has $20 of OpenRouter credit and wants breadth and curiosity over thrift).

## Status (2026-10-13)

**Probe results (live, about $0.003 in all):**
- **Transport A works.** The Anthropic 1.x SDK with `base_url="https://openrouter.ai/api"` returned schema-valid JSON through `output_config.format` for `anthropic/claude-haiku-5.5` and `deepseek/deepseek-v4-flash`, with `thinking` (adaptive, summarized) and a system `cache_control` accepted, and `usage.cost` (USD) on every reply. `provider` (`require_parameters`, `data_collection`, `max_price`) and `openrouter/auto`'s `plugins` + `session_id` pass through `extra_body`. An unmet `max_price` and an empty auto candidate set are both `404 not_found_error`. **So there is no new model class**: `AnthropicWorkflowModel` runs on OpenRouter with a different client, `extra_body`, and `reported_cost`. Transport B was not needed.
- **Auto** with `allowed_models = ["anthropic/*"]`, `medium`, picked `anthropic/claude-sonnet-5.5`; with `deepseek/*, z-ai/*, mistralai/*`, `low`, picked `deepseek/deepseek-v4-flash-0731`. Both honoured `require_parameters`.
- **Catalog:** 461 models, 380 with `structured_outputs`, only **3 free** ones that do (`dots-studio/dots-3-note-preview:free`, `liquid/lfm-2.5-2.6b:free`, `nvidia/nemotron-3-super-120b-a12b:free`). A `:free` request with `data_collection: "deny"` is a `404 No endpoints found matching your data policy`, as predicted.
- **Ids use dots, not dashes:** `anthropic/claude-haiku-5.5`, not `claude-haiku-5-5`. The catalog also lists `:batch` variants as separate ids.
- **The key has a $5 credit limit** (`limit: 5`, set on the key, not the account), and 1,000 free requests a day.

**Built (no live runs of Workflows yet):**
- `--backend openrouter`, and any `vendor/model` id picks it (`choose`); `openrouter_model` in `[workflow]` is the default model.
- `[openrouter]` config; `openrouter_body` builds the routing fields; `make_openrouter_client` reads `OPENROUTER_API_KEY` and nothing else.
- `usage.cost` is the call's cost (`api_call(cost_usd=...)`); a reply without it fails (`UnreportedCost`) instead of counting $0.
- Preflight (`prepare_openrouter`): the key, every model in the catalog with structured output, an `openrouter/auto` allow-list of such models, and the key's remaining credit. Exits 2 before any call.
- Per reply: a served model without structured output fails the call (`UnstructuredModel`); an unknown served id passes and is recorded.
- Roles and the Arbiter accept OpenRouter ids.
- `harness-workflow models [pattern] [--free] [--max-price] [--all] [--json]` over the cached catalog; `bench/model-notes.toml` ranking notes.
- Bench accepts OpenRouter cells; `--batch` is refused with them.
- Gates 1–5, 7–10, 12–13 have pytest tests (`tests/workflow/test_openrouter_model.py`, `tests/test_openrouter_catalog.py`).

**Not built yet:** the per-call `provider`, `task_type` and `cost_tier` records and the `X-OpenRouter-Metadata` header (only `served_model` is recorded today); `session_id` is a random id per model object, not the run id; the `bench/model-notes.toml` ranking notes file ships empty (the `models` command reads it when you fill it in); `ApiFailure`/`API_FAULTS` still live in `anthropic_api.py`; Plan provenance and `plans-for` / `report --pairs`; `report --frontier` served-model histogram and `~` marking for auto; a Spend cap check against the key's remaining credit; the live Workflow checks below.

## What OpenRouter gives us (checked 2026-10-13 against its docs)

| Need | OpenRouter feature |
|---|---|
| One key, many models | `https://openrouter.ai/api/v1`, bearer auth, model ids like `vendor/model` |
| Free models | A `:free` suffix on a model id that has a free entry. 20 requests/min on every account. 50 requests/day until you have bought ≥10 credits in total, then 1,000/day. Extra keys or accounts do not raise it. `GET /api/v1/key` reports `free_model_daily_requests` used and remaining |
| Cheapest provider for a model | `:floor` suffix (sort by price), `:nitro` (throughput); `provider.max_price` hard-caps the price and the request fails rather than exceed it |
| Fallback chain | `models: [a, b, c]`: any error (rate limit, downtime, context length, moderation) moves to the next; **billed at the model that answered**, named in the response `model` field |
| Let it choose | `openrouter/auto` classifies the prompt and picks a model by `cost_tier`; `allowed_models`/`excluded_models` (wildcards such as `anthropic/*`) bound it; no router fee; the chosen model is in `model` |
| Structured output | `response_format: {type: json_schema, strict: true}`. Enforcement varies by model and provider, so send `provider.require_parameters: true` or a provider that ignores the schema will silently reply with free text |
| Cost per call | `usage.cost` (credits = USD) in the response, plus `prompt_tokens_details.cached_tokens` / `cache_write_tokens`; `/api/v1/generation` later |
| Privacy | `provider.data_collection: "deny"` and `provider.zdr: true`. **Free models often log prompts for training**; treat them as public |
| Anthropic-shaped endpoint | OpenRouter documents `ANTHROPIC_BASE_URL=https://openrouter.ai/api` for Claude Code. Whether the 1.x SDK's `output_config.format`, `thinking` and `cache_control` work through it for non-Claude models is **unverified**: see the probe |

## Decisions to make (recommendations)

1. **Transport.** Two ways to reach it:
   - **A. The Anthropic SDK with `base_url="https://openrouter.ai/api"`.** `AnthropicWorkflowModel` and `AnthropicChatModel` (streaming, tools) would work as they are, with a `make_client` that takes a base URL and key. Cheapest by far, if the probe passes.
   - **B. OpenAI-style `/chat/completions` through `httpx2`** (no new dependency, same `httpx2.MockTransport` fakes). A new `OpenRouterWorkflowModel` beside the other two, behind the same `WorkflowModel` seam.
   - *Superseded by the probe (see Status): `plugins`, `session_id` and `provider` all pass through the Anthropic-shaped endpoint, so A covers `openrouter/auto` too and B was not built.*
   - **Recommend:** run the probe (below, about $0), then take A if a non-Claude model returns schema-valid JSON through it, else B. B is the safe default for non-Claude models whatever the probe says, because `require_parameters` and `models`/`provider` routing are first-class there. **Workflows first; the chat agent stays on Anthropic** until a Workflow row says another model is worth it.
2. **Free models are a bench variant, never the default.** At 20/min and 1,000/day (the $20 of credit should clear the $10 purchase threshold: check `/api/v1/key` in the probe) a free model can carry a few katas, not a 108-run sweep, and can be rate-limited into a `429` mid-run. Good for sampling a new model for nothing.
3. **`openrouter/auto` is a first-class choice, treated as a measured Cell.** It is a model id like any other (`--model openrouter/auto`), but what a run measures is "auto at this tier, with this allow-list, on this date", not a model: its ranking window moves weekly, so a re-run can route differently. Every call's served model and task type are recorded, and the report shows the distribution (see "openrouter/auto" below). Pin a model when you want a repeatable row.
4. **Roles are the right unit for routing.** F11 already gives the tester, implementer and **Arbiter** their own model, so "model A tests, model B implements, model C arbitrates" is `[workflow.roles.*]`, not a router.
5. **Refuse a model without structured output** (decided 2026-10-13). A model whose `supported_parameters` lacks `structured_outputs` is not a Workflow model; the bench and `plan`/`build` exit 2 naming it, before any call.

## Shape

- **Config** (`harness.toml`):
  ```toml
  [openrouter]
  max_retries = 2
  timeout = 300.0
  data_collection = "deny"        # sent as provider.data_collection; "allow" needed for most :free models
  require_parameters = true       # never route to a provider that ignores response_format
  max_price = { prompt = 1.0, completion = 3.0 }   # USD per million; a request over it fails instead of running
  fallbacks = []                  # ["vendor/other-model"]: the `models` chain
  # No API key here, ever: read OPENROUTER_API_KEY from the environment.
  ```
  `[workflow] backend` and `--backend` gain `openrouter`. A model id containing `/` picks it (Claude ids keep `claude-` → Anthropic; `anthropic/claude-…` goes through OpenRouter only when `--backend openrouter` is explicit). `[workflow.roles.*] model` accepts an OpenRouter id (today they must start with `claude-`).
- **Pricing.** `PRICING` is a static table; OpenRouter has hundreds of models. Use `usage.cost` from each response as the truth, and read `GET /api/v1/models` once per run (cached in `~/.braid/openrouter-models.json`, 24 h) for each model's per-token prices and `supported_parameters`. The **Spend cap** uses the response's `cost` when present and the cached prices otherwise. A `:free` model costs $0 and counts $0 toward the cap.
- **Preflight (API flavour, runs before the first call).** The model exists; `structured_outputs` is in its `supported_parameters` (else exit 2 naming the model, decision 5); `GET /api/v1/key` shows credit left and, for a `:free` model, `free_model_daily_requests.remaining`. A run that needs more calls than remain is refused before it starts, not stopped halfway.
- **Failures.** `429`, `402` (key credit limit), `5xx`, a provider error inside a `200`, and `finish_reason: "error"` become `ApiFailure` kinds, so the run ends `api_error`, resumable, as in F8. The key pattern `sk-or-…` is added to the redaction in `anthropic_api.py` (or its successor).
- **Records.** Every model-node log line adds `provider` (the upstream that served it) and `served_model` (response `model`), alongside tokens, `cost_usd` and `request_id` (the OpenRouter generation id). The bench summary groups cost by served model; the frontier report (F12) treats a label as `model + routing` so a `:free` row and its paid twin sit side by side.
- **Sandbox.** `OPENROUTER_API_KEY` joins `ANTHROPIC_API_KEY` in the env scrub (ADR 0005); a gate asserts it.
- **Guard.** `guard/` never imports the HTTP client for this (ADR 0001, 0006).

## openrouter/auto

How it works (docs, 2026-10-13): a classifier puts each prompt in one of ~30 task types (`code:debugging`, `math`, ...); candidates are ranked by their **share of OpenRouter spend for that task over the last 7 days**; `cost_tier` (`low` … `max`, sent in the plugin, default roughly `low`) is a *band* that excludes models below it as well as above; the top few become the primary pick and its fallbacks. No router fee. `allowed_models` / `excluded_models` (wildcards) bound it, and an empty result is a `404`. The reply's `model` names who answered; `X-OpenRouter-Metadata: enabled` adds `task_type`.

**What that means for braid**

- **It ranks by popularity, not quality, and not novelty.** Spend share favours the models many people already use for a task. It will tell you "what does the crowd pay for on code tasks" well; it will rarely surface the new-to-me model. For discovery, pin ids or sweep a list (`bench --models`); use auto as the *baseline the crowd would pick*, to beat or match.
- **Each node may route differently.** braid's calls are single-turn with distinct system prompts (critique angles, tester, implementer, Polish), so one run can mix several models. That is an emergent version of F11 Roles, free of charge, but it also breaks prompt-cache continuity. Mitigation: send `session_id = run id`, so the router keeps the earlier model while it is still a top candidate (set `session = "none"` to see unconstrained routing).
- **Structured output is the risk.** The docs do not say candidates are filtered by `response_format`. A top-ranked model without `structured_outputs` would answer in free text. Two guards, both required: send `provider.require_parameters: true`, and treat a reply whose served model lacks `structured_outputs` in the cached `/api/v1/models` as an `api_error`-class failure, not a parse retry. **Decision 5 therefore works per reply for auto**; Preflight can only vet the allow-list, so `allowed_models` must be a list of exact ids (or wildcards that `models --json` shows are all JSON-capable) before a bench is allowed to use auto.
- **The tier and the allow-list interact.** An allow-list of models all below the tier gives a `404`. Preflight prints the allow-list's models inside the tier and refuses an empty intersection.
- **Cost is unknowable beforehand.** The Spend cap still holds at node boundaries because every reply carries `usage.cost`; `provider.max_price` and the tier bound a single call. The probe prints the worst-case price in the tier's band.
- **Planner and executor.** Auto is two separate Cells: an *auto planner* and an *auto executor*. A restricted auto is a cheap way to say "Anthropic plans, anyone good codes": planner `allowed_models = ["anthropic/*"]`, executor `allowed_models = ["qwen/*", "deepseek/*", "mistralai/*"]` (the families to try), tier chosen so the band holds the models you mean.

**Config**

```toml
[openrouter.auto]
cost_tier = "medium"                 # low | medium | high | xhigh | max; the band, not a ceiling
allowed_models = ["anthropic/*"]     # wildcards or exact slugs; checked at Preflight
excluded_models = []
session = "run"                      # "run": session_id = run id (sticky); "none"
metadata = true                      # X-OpenRouter-Metadata: enabled, to record task_type
# per Role: [workflow.roles.tester] model = "openrouter/auto" uses [openrouter.auto.tester] ... if present
```

**Records.** Each model-node line adds `routed_by = "openrouter/auto"`, `task_type`, `served_model` and `cost_tier`. The summary and `bench` report show a histogram of served models per Cell (for example `qwen3-coder 61%, claude-sonnet 25%, ...`) and the cost by served model; `report --pairs` marks auto Cells with `~` and the date, because they are not repeatable. These are OpenRouter's routing decisions, not graph edges, so they are outside ADR 0008's Route records, but they get the same treatment: nothing a router chooses is hidden.

## Pairing: one model plans, another codes

The two loops are already separate commands with their own `--model`, so a pairing needs no graph change:

```
# Claude plans (a Plan is the hand-off, saved in ~/.braid/plans/<id>.json)
harness-workflow plan "<goal>" --backend openrouter --model anthropic/claude-sonnet-5-5
# another model executes it
harness-workflow build --plan ~/.braid/plans/<id>.json --backend openrouter --model qwen/qwen3-coder
```

What is new is making the pairing a measured thing:

- **Plan provenance.** A saved Plan records the model, effort and backend that wrote it (check `workflow/plan.py`; add the field if absent). The bench summary carries `planner` next to `executor`.
- **Bench sets from a planner.** `bench` already runs a named set of (kata, Plan) pairs (`claude-plans`, `local-plans`). Add `harness-workflow plans-for <set> --model <id>` to write a Plan per kata with a given planner and register them as a set named `<planner>-plans`, so any executor can run on them. The planning cost is recorded.
- **Planner × executor grid.** `bench --plans a-plans,b-plans --models x,y` is the cross product; `report --pairs` prints hidden pass rate as a matrix (rows planner, columns executor), with cost per cell. This answers "does a stronger planner lift a weaker coder, and does a strong coder rescue a weak planner?", which F1's live check began with Claude and e4b Plans.
- **Rankings.** Keep a hand-edited `bench/model-notes.toml`, `"vendor/model" = { index = 61, source = "artificialanalysis.ai", as_of = "2026-10" }`. `models` and `report --pairs` show it as a column, so published intelligence rankings sit next to what the katas measured. It is never fetched (no ranking site's API has been checked); the value is yours to paste in.

## Trying a new model, and free ones

```
# 1. See what is free and what it supports (JSON-capable only), cheapest first
harness-workflow models --free
harness-workflow models --json --max-price 0.5      # structured-output models under $0.50/M input

# 2. Try a new model on the five katas, with a hard cap
harness-workflow bench claude-plans --backend openrouter --model qwen/qwen3-coder:free --repeat 1 --label try-qwen --max-cost 0.05

# 3. Compare against the Haiku baseline you already have
harness-workflow report --frontier baseline-f1 try-qwen
```

- `models` is a small command over the cached `/api/v1/models`: id, context window, input/output $/M, `free`, `structured_outputs`, `tools`. No spend.
- `bench --models a,b,c` (F12) already sweeps a grid; with this backend a Cell may be an OpenRouter id, so "five new models × the katas" is one command, and `report --frontier` ranks hidden pass rate against cost. That is how "most bang for the buck" becomes a number, not a guess.
- To keep a lid on price: append `:floor` to a paid id, set `max_price`, and let `fallbacks` cover a free model's rate limit. To let it choose: `--model openrouter/auto` (see its section).
- **Budget.** The Spend cap stays per run (default $0.10, raise it for big models). Add `--max-cost` per bench as now, and have `models`/Preflight print the key's remaining credit, so $20 does not vanish into one sweep unnoticed.

## Gates (`tests/workflow/test_openrouter_model.py`, `httpx2.MockTransport`, no network)

1. A valid structured reply parses; an invalid one then a valid one parses after exactly two requests; two invalid raise `WorkflowModelError`.
2. The request carries `response_format.json_schema` with `strict`, `provider.require_parameters: true`, the configured `data_collection` and `max_price`, and no `temperature`/`top_p`/`top_k` (the default model may reject them; sampling is a later knob).
3. `usage.cost` becomes `cost_usd`; with no `cost`, the cached model prices compute it; a `:free` model records $0.
4. The `model` field of a reply is recorded as `served_model`; after a `fallbacks` hop the log shows the served model, not the first.
5. `429`, `402`, an error body inside a `200`, and `finish_reason: "error"` each end the run `api_error` (resumable); none is retried by node code.
6. Preflight refuses a model without `structured_outputs`, and refuses a `:free` run when `remaining` is below the estimated calls. Both exit before any completion request.
7. The key appears in no run log, summary, error text or Sandbox environment.
8. A Spend cap stops an OpenRouter run at the next node boundary with `spend_cap`, nothing applied.
9. `guard/` imports nothing for OpenRouter (import-graph test).
10. `models --free` filters and prints from a fixture of `/api/v1/models`.
11. A Plan records its planner; a planner × executor bench writes both to the summary, and `report --pairs` prints the matrix from fixtures, with a note column from `model-notes.toml`.
12. A model without `structured_outputs` exits 2 from `plan`, `build` and `bench` before any completion request.
13. `openrouter/auto`: the request carries the `auto-router` plugin with `cost_tier`, `allowed_models`, `excluded_models`, `session_id` = run id (or none), `provider.require_parameters: true` and the metadata header; the reply's `model` and `task_type` are recorded; a served model whose cached `supported_parameters` lacks `structured_outputs` fails the call as an error, not a retry; a `404` for an empty candidate set ends the run `api_error` with that reason.
14. Preflight for auto refuses an `allowed_models` that matches no JSON-capable model inside the tier, and a bench Cell using auto without an allow-list.
15. The report's served-model histogram and `~` marking come out of fixtures for a run that was routed to three models.

## Live checks (spend approved first)

1. **Probe, ~$0.** One request each to `claude-haiku-5-5` via `anthropic/claude-haiku-5-5`, a cheap paid non-Claude model, and one `:free` model: through transport A (Anthropic SDK, `base_url`) and transport B. Record which return schema-valid JSON, whether `thinking`/`cache_control` pass, whether `usage.cost` is present. This decides the transport.
2. **Katas on 2–3 new models,** `--repeat 1`, `--no-hints`, against `baseline-f1` and `baseline-exotic`, one free and one or two cheap paid. Expect cents. Report hidden pass rate, cost, and the **self/hidden gap**.
3. **Pairings:** Claude (Sonnet 5.5) plans the five katas once; two or three other coders (one free) execute those Plans against Haiku's `baseline-f1` row; then the reverse for one kata (a cheap model plans, Claude codes). Report the matrix.
4. **Auto probe, ~$0.01:** `openrouter/auto`, `cost_tier = "medium"`, `allowed_models = ["anthropic/*"]`, one tiny schema request, with and without `require_parameters`. Record: does it ever pick a model without `structured_outputs`; does `plugins` pass on the Anthropic-shaped endpoint (expected no); `task_type` from the metadata header; the models in each tier's band.
5. **Auto on the katas:** auto planner (Anthropic-only allow-list) → auto executor (a few families), `--repeat 1`, against the pinned Sonnet-plans × Haiku row. Report the served-model histogram next to the pass rate.
6. **A strong Arbiter** via `[workflow.roles.arbiter]` on one disputed Plan.

## Effort (measured 2026-10-13) and a later effort chooser

`output_config.effort` is honoured through OpenRouter on `deepseek/deepseek-v4-flash:nitro` (thinking on, one bowling-scorer prompt): **low** 509 thinking tokens in 21 s; none sent 1,649 in 43 s; **medium** 2,785 in 70 s; **high** 4,947 in 140 s. Thinking disabled: 0 in 3.5 s. Braid sends no effort unless `[workflow] effort` or `bench --effort` is set, so the first OpenRouter bench ran at the model's default. The plan is to run the katas at `low` first and let the sweep (`--effort low,medium`) say whether more is worth it.

**Effort chooser (idea, not built).** A small decision model of our own, not OpenRouter's router, picks the effort (and whether thinking is on) per node from cheap signals: the node (write_test, write_impl, polish), the attempt number, the last failure. It fits ADR 0008 if it is a schema field, is recorded as a Route record, is bounded (a rung up per failed attempt at most), and never declares work done. It would sit beside the existing advisory **Decision layer** (ADR 0003). Start rule-based (low first; step up one effort after a failed attempt), measure it against fixed low and fixed medium, and only then try a model.

## Out of scope

The chat agent on OpenRouter; Batches (OpenRouter has none); prompt-caching tuning beyond recording `cached_tokens`; BYOK; Ollama.

## Proposed ADR 0009 (to write before code)

*"OpenRouter is a third backend, behind the `WorkflowModel` seam."* Amends ADR 0007's deferral. Consequences to record: data goes to OpenRouter **and** the upstream provider (free models may train on it, so `data_collection = "deny"` is the default and free models are for non-sensitive katas only); routing by `openrouter/auto` is OpenRouter's choice, ranked by spend share, not a graph edge: allowed anywhere a model id is, but a run on it is a dated, labelled measurement with the served model, task type and tier recorded (ADR 0008's trace rule in spirit), and structured output is checked per reply because the router is not documented to filter on it; prices come from the response, not `PRICING`; structured output is verified at Preflight, never assumed.

## Open questions

- Is the user's $20 credit *purchased* (1,000 free requests a day) or granted? The probe reads `/api/v1/key`.
- Does `openrouter/auto` filter candidates by `response_format`? The docs are silent; the auto probe answers it, and decides whether per-reply refusal is a safety net or the main path.
- Which models to try first? The probe can print the structured-output models on OpenRouter for the user to choose from, with their ranking notes.

Settled 2026-10-13: models without structured output are refused (decision 5).
