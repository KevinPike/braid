"""Budget math: the largest num_ctx that fits the GPU limit."""

from __future__ import annotations

from harness.guard.ollama_client import ModelInfo

GIB = 1024**3
CTX_STEP = 1024


def kv_bytes_upper_bound(*, layers: int, kv_heads: int, head_dim: int, bytes_per_element: int) -> int:
    """Theoretical KV bytes per token; an upper bound for sliding-window models (ADR 0002)."""
    return 2 * layers * kv_heads * head_dim * bytes_per_element


SWA_BATCH_CELLS = 512  # llama.cpp sizes the sliding-window cache at window + ubatch


def _owning(info: ModelInfo) -> list[tuple[int, bool]]:
    """(kv_heads, is_sliding) for each layer that owns a KV cache; shared trailing layers own none."""
    pattern = info.sliding_pattern or (False,) * info.layers
    owning = info.layers - info.shared_kv_layers
    return [(info.kv_heads[i], pattern[i]) for i in range(owning)]


def kv_bytes_per_token(info: ModelInfo, bytes_per_element: int = 2) -> int:
    """KV bytes that grow with num_ctx: global-attention layers only.

    Verified against the Ollama server log for gemma4:e4b and gemma4:12b (16384 B/token each).
    """
    heads = sum(h for h, swa in _owning(info) if not swa)
    return heads * (info.key_length + info.value_length) * bytes_per_element


def kv_fixed_bytes(info: ModelInfo, bytes_per_element: int = 2) -> int:
    """KV bytes independent of num_ctx: sliding-window layers hold a fixed window."""
    heads = sum(h for h, swa in _owning(info) if swa)
    cells = info.sliding_window + SWA_BATCH_CELLS
    return heads * cells * (info.key_length_swa + info.value_length_swa) * bytes_per_element


def max_ctx(
    *,
    gpu_limit: int,
    margin: int,
    helpers: int,
    weights: int,
    overhead: int,
    kv_per_token: float,
    trained_ctx: int,
) -> int:
    """Budget: floor((limit - margin - helpers - W - O) / k), rounded down to 1024, capped at trained."""
    room = gpu_limit - margin - helpers - weights - overhead
    if room <= 0 or kv_per_token <= 0:
        return 0
    raw = min(int(room // kv_per_token), trained_ctx)
    return raw // CTX_STEP * CTX_STEP
