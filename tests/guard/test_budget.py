from harness.guard.budget import GIB, kv_bytes_per_token, kv_bytes_upper_bound, kv_fixed_bytes, max_ctx
from harness.guard.ollama_client import ModelInfo


def test_max_ctx_rounds_down_to_1024_and_caps_at_trained() -> None:
    # (16 - 1.5 - 0 - 8 - 0.75) GiB = 5.75 GiB of KV room at 100 KiB/token
    k = 100 * 1024
    got = max_ctx(
        gpu_limit=16 * GIB, margin=int(1.5 * GIB), helpers=0,
        weights=8 * GIB, overhead=int(0.75 * GIB), kv_per_token=k, trained_ctx=1_000_000,
    )
    assert got == 59392  # floor(5.75 GiB / 100 KiB) = 60293 -> 58 * 1024
    assert got % 1024 == 0
    capped = max_ctx(
        gpu_limit=16 * GIB, margin=int(1.5 * GIB), helpers=0,
        weights=8 * GIB, overhead=int(0.75 * GIB), kv_per_token=k, trained_ctx=32768,
    )
    assert capped == 32768


def test_max_ctx_zero_when_weights_do_not_fit() -> None:
    assert max_ctx(
        gpu_limit=8 * GIB, margin=GIB, helpers=0, weights=10 * GIB,
        overhead=GIB // 2, kv_per_token=1024, trained_ctx=32768,
    ) == 0


def test_kv_upper_bound_formula() -> None:
    # 2 * layers * kv_heads * head_dim * bytes
    assert kv_bytes_upper_bound(layers=32, kv_heads=8, head_dim=128, bytes_per_element=2) == 131072


def e4b_info() -> ModelInfo:
    # Architecture fields from `ollama show gemma4:e4b`; True = sliding-window layer.
    return ModelInfo(
        layers=42, kv_heads=(2,) * 42, key_length=512, value_length=512, trained_ctx=131072,
        sliding_pattern=tuple(i % 6 != 5 for i in range(42)), shared_kv_layers=18,
        sliding_window=512, key_length_swa=256, value_length_swa=256,
    )


def test_architecture_aware_kv_matches_server_log_for_gemma4_e4b() -> None:
    # Server log: non-SWA cache 32 MiB at 2048 cells (4 layers) -> 16384 B/token;
    # SWA cache 40 MiB for 20 layers, independent of num_ctx.
    info = e4b_info()
    assert kv_bytes_per_token(info) == 16384
    assert kv_fixed_bytes(info) == 40 * 1024 * 1024


def test_dense_model_counts_every_layer_as_global() -> None:
    dense = ModelInfo(layers=32, kv_heads=(8,) * 32, key_length=128, value_length=128, trained_ctx=8192)
    assert kv_bytes_per_token(dense) == 131072
    assert kv_fixed_bytes(dense) == 0


def test_per_layer_kv_heads_match_server_log_for_gemma4_12b() -> None:
    # `ollama show gemma4:12b`: head_count_kv is per layer (8 on sliding layers, 1 on global).
    # Server log: non-SWA 64 MiB at 4096 cells (8 layers) -> 16384 B/token;
    # SWA 480 MiB at 1536 cells (40 layers).
    pattern = tuple(i % 6 != 5 for i in range(48))
    info = ModelInfo(
        layers=48, kv_heads=tuple(8 if swa else 1 for swa in pattern), key_length=512, value_length=512,
        trained_ctx=262144, sliding_pattern=pattern, sliding_window=1024, key_length_swa=256, value_length_swa=256,
    )
    assert kv_bytes_per_token(info) == 16384
    assert kv_fixed_bytes(info) == 480 * 1024 * 1024
