import json

import httpx
import pytest
import respx

from harness.guard.ollama_client import CallMetrics, OllamaClient, RunningModel, UnpinnedModelError, UnsupportedModelError

HOST = "http://ollama.test"

FINAL = {
    "model": "m", "done": True, "message": {"role": "assistant", "content": ""},
    "prompt_eval_count": 120, "eval_count": 40, "eval_duration": 2_000_000_000,
    "load_duration": 5_000_000, "total_duration": 3_000_000_000,
}


def ndjson(*objs: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, content="\n".join(json.dumps(o) for o in objs).encode())


@pytest.fixture
def client() -> OllamaClient:
    c = OllamaClient(HOST, keep_alive="30m")
    c.pin("m", 8192)
    return c


@respx.mock
async def test_chat_pins_num_ctx_and_keep_alive_and_reports_metrics(client: OllamaClient) -> None:
    route = respx.post(f"{HOST}/api/chat").mock(
        return_value=ndjson({"message": {"role": "assistant", "content": "hi"}, "done": False}, FINAL)
    )
    seen: list[tuple[str, CallMetrics]] = []
    client.on_call(lambda model, m, est: seen.append((model, m)))

    chunks = [c async for c in client.chat("m", [{"role": "user", "content": "yo"}])]

    body = json.loads(route.calls.last.request.content)
    assert body["options"]["num_ctx"] == 8192
    assert body["keep_alive"] == "30m"
    assert body["stream"] is True
    assert "".join(c.text for c in chunks) == "hi"
    assert seen == [("m", CallMetrics(prompt_eval_count=120, eval_count=40, eval_duration_s=2.0, load_duration_s=0.005))]


@respx.mock
async def test_caller_options_cannot_override_pinned_num_ctx(client: OllamaClient) -> None:
    route = respx.post(f"{HOST}/api/chat").mock(return_value=ndjson(FINAL))
    _ = [c async for c in client.chat("m", [], options={"num_ctx": 999, "temperature": 0.2})]
    opts = json.loads(route.calls.last.request.content)["options"]
    assert opts == {"num_ctx": 8192, "temperature": 0.2}


async def test_unpinned_model_is_refused(client: OllamaClient) -> None:
    with pytest.raises(UnpinnedModelError):
        _ = [c async for c in client.chat("other", [])]


@respx.mock
async def test_unload_sends_keep_alive_zero(client: OllamaClient) -> None:
    route = respx.post(f"{HOST}/api/generate").mock(return_value=httpx.Response(200, json={"done": True}))
    await client.unload("m")
    assert json.loads(route.calls.last.request.content) == {"model": "m", "keep_alive": 0}


@respx.mock
async def test_ps_and_version(client: OllamaClient) -> None:
    respx.get(f"{HOST}/api/version").mock(return_value=httpx.Response(200, json={"version": "0.35.1"}))
    respx.get(f"{HOST}/api/ps").mock(return_value=httpx.Response(200, json={"models": [{
        "name": "m", "digest": "d1", "size": 100, "size_vram": 90,
        "expires_at": "2026-10-08T09:22:47.768035-07:00", "context_length": 8192,
    }]}))
    assert await client.version() == "0.35.1"
    (rm,) = await client.ps()
    assert (rm.name, rm.size, rm.size_vram, rm.context_length) == ("m", 100, 90, 8192)
    assert rm.spilled and rm.gpu_fraction == 0.9
    assert rm.expires_at.year == 2026


@respx.mock
async def test_show_extracts_architecture(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/show").mock(return_value=httpx.Response(200, json={
        "details": {"quantization_level": "Q4_K_M"},
        "capabilities": ["completion"],
        "model_info": {
            "general.architecture": "gemma4", "gemma4.block_count": 42,
            "gemma4.attention.head_count": 8, "gemma4.attention.head_count_kv": 2,
            "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 512, "gemma4.attention.shared_kv_layers": 18,
            "gemma4.attention.sliding_window_pattern": [True, True, False],
            "gemma4.context_length": 131072, "gemma4.embedding_length": 2560,
        },
    }))
    info = await client.show("m")
    assert (info.layers, info.kv_heads, info.key_length, info.value_length, info.trained_ctx) == (42, (2,) * 42, 512, 512, 131072)
    assert (info.shared_kv_layers, info.sliding_window, info.key_length_swa) == (18, 512, 256)
    assert info.sliding_pattern == (True, True, False)


def test_ps_undercount_detection() -> None:
    """ollama#17251: ps reports the speculative draft's footprint instead of the target's."""
    from datetime import datetime, timezone

    def entry(size: int) -> "RunningModel":
        return RunningModel("m", "d", size, size, datetime.now(timezone.utc), 4096)

    assert not entry(7_000).undercounts(expected_size=10_000)
    assert entry(1_000).undercounts(expected_size=10_000)
    assert not entry(1_000).undercounts(expected_size=0)  # unknown expectation never flags



@respx.mock
async def test_show_accepts_per_layer_kv_head_list(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/show").mock(return_value=httpx.Response(200, json={"model_info": {
        "general.architecture": "g", "g.block_count": 3, "g.attention.head_count": 16,
        "g.attention.head_count_kv": [8, 8, 1], "g.context_length": 1024, "g.embedding_length": 64,
    }}))
    assert (await client.show("m")).kv_heads == (8, 8, 1)


@respx.mock
async def test_show_treats_null_optional_fields_as_absent(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/show").mock(return_value=httpx.Response(200, json={"model_info": {
        "general.architecture": "g", "g.block_count": 3, "g.attention.head_count": 16,
        "g.context_length": 1024, "g.embedding_length": 64,
        "g.attention.sliding_window_pattern": None, "g.attention.shared_kv_layers": None,
        "g.attention.sliding_window": None,
    }}))
    info = await client.show("m")
    assert (info.sliding_pattern, info.shared_kv_layers, info.sliding_window) == ((), 0, 0)


@respx.mock
async def test_show_without_attention_fields_is_unsupported(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/show").mock(return_value=httpx.Response(200, json={"model_info": {
        "general.architecture": "g", "g.block_count": 3, "g.context_length": 1024, "g.embedding_length": 64,
    }}))
    with pytest.raises(UnsupportedModelError, match="KV cost"):
        await client.show("m")


@respx.mock
async def test_show_requests_verbose_so_per_layer_arrays_are_not_elided(client: OllamaClient) -> None:
    route = respx.post(f"{HOST}/api/show").mock(return_value=httpx.Response(200, json={"model_info": {
        "general.architecture": "g", "g.block_count": 3, "g.attention.head_count": 16,
        "g.context_length": 1024, "g.embedding_length": 64,
    }}))
    await client.show("m")
    assert json.loads(route.calls.last.request.content)["verbose"] is True
