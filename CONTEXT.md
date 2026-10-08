# Local Agent Harness

A terminal agent harness that runs local Ollama models on a 24 GB Mac and keeps context use and runtime health visible and guarded. It exists to surface what Ollama hides.

## Language

### Runtime health

**Guard**:
The standalone `guard/` module that chooses a safe context length before a model loads and watches every call afterwards. Knows nothing about Strands.
_Avoid_: Monitor, supervisor

**Preflight**:
The loud, fail-fast checklist run before a model loads; ends with a **Budget** check and a confirmed full-GPU load.
_Avoid_: Startup check, health check

**Watchdog**:
The in-session half of the **Guard**: a background `/api/ps` poller plus per-call metric checks, each raising an **Alert**.
_Avoid_: Poller, health monitor

**Alert**:
A watchdog finding with a severity (yellow or red) shown in the status bar or banner.
_Avoid_: Warning, error

**Budget**:
The largest `num_ctx` that fits the GPU limit after margin, helpers, weights and overhead.
_Avoid_: Limit, quota

**Calibration**:
Measuring a model's KV bytes per token by loading it at two context lengths, cached per Ollama version and model digest.
_Avoid_: Estimation, benchmarking

**Pinned num_ctx**:
The single `num_ctx` a model runs at for a whole session, set only by the one client every request goes through.
_Avoid_: Context size, context setting

### Failure modes

**Spill**:
A loaded model with `size_vram < size`, meaning part of it runs on CPU.
_Avoid_: Offload, fallback

**Truncation**:
Ollama silently dropping prompt tokens because the prompt exceeded `num_ctx`; detected when `prompt_eval_count` falls below the harness estimate.
_Avoid_: Overflow, clipping

**Reload**:
Ollama loading a model again mid-session, usually from `num_ctx` drift or eviction.
_Avoid_: Restart

### Context

**Context window**:
The pinned `num_ctx` tokens a call may use; split into categories (system prompt, tool schemas, history, tool results, current turn).
_Avoid_: Prompt size, memory

**Compaction**:
Shrinking history to stay under the pressure thresholds, by sliding window, summary or offloading; every step is recorded in the **Trim log**.
_Avoid_: Truncation (reserved for the failure mode), pruning

**Trim log**:
The SQLite record of what was dropped or summarized, when and why.
_Avoid_: Compaction history

### Models

**Daily driver**:
The model that runs the main loop (`gemma4:e4b`).
_Avoid_: Main model, default model

**Decision layer**:
The tev1 models that answer routing, compaction-trigger, policy and done-or-continue questions. Advisory only; never the sole check on a risky action.
_Avoid_: Router, classifier

**Profile**:
The set of models, roles and starting `num_ctx` values in `harness.toml`.
_Avoid_: Preset, config (config is the whole file)

### Process

**Milestone**:
A numbered, zero-indexed unit of the plan (M0–M10) that ships something usable and has a validation gate that must pass before the next starts.
_Avoid_: Phase, sprint

**Validation gate**:
The checkable criteria that close a **Milestone**.
_Avoid_: Acceptance test
