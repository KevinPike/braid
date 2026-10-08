# Measure KV cost per model instead of computing it

Budget math uses KV bytes per token fitted from two real loads, cached per Ollama version and model digest. The architecture formula (`2 * layers * kv_heads * head_dim * bytes`) is only an upper bound: Gemma's sliding-window and shared-KV layers use far less, so trusting it would under-use the GPU. Calibration is the source of truth; the formula is a cross-check (calibrated k within 25% of it, or the gap explained).

## Update (M1): calibration is blocked by ollama#17251

On Ollama 0.35.1 the `gemma4` models load a speculative draft model, and `/api/ps` reports the draft's footprint instead of the target's ([ollama/ollama#17251](https://github.com/ollama/ollama/issues/17251); fix in PR #17857, unmerged). Two-load calibration from `ps` `size` then measures the wrong model, so `Calibrator` refuses when `ps` size is under half the model's on-disk size. Until upstream is fixed, budgets use an architecture-derived KV cost (global-attention layers per token, sliding-window layers as a fixed cost, shared-KV layers free), verified against the server log for `gemma4:e4b` and `gemma4:12b`. Calibration remains the source of truth wherever `ps` is trustworthy.
