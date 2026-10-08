# Measure KV cost per model instead of computing it

Budget math uses KV bytes per token fitted from two real loads, cached per Ollama version and model digest. The architecture formula (`2 * layers * kv_heads * head_dim * bytes`) is only an upper bound: Gemma's sliding-window and shared-KV layers use far less, so trusting it would under-use the GPU. Calibration is the source of truth; the formula is a cross-check (calibrated k within 25% of it, or the gap explained).
