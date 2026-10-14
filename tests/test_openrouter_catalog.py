"""The OpenRouter catalog, key status and model listing (specs/openrouter.md); httpx2.MockTransport, no network."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx2
import pytest

from harness.openrouter import (
    AUTO,
    Catalog,
    ModelRefused,
    OpenRouterError,
    key_status,
    load_catalog,
    load_notes,
    parse_catalog,
    refuse_unstructured,
    render_models,
    vet_allow_list,
)

FULL = ["tools", "structured_outputs", "response_format", "temperature"]


def entry(model_id: str, prompt: str, completion: str, params: list[str]) -> dict[str, Any]:
    return {
        "id": model_id, "context_length": 200000, "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": params,
    }


PAYLOAD: dict[str, Any] = {"data": [
    entry("anthropic/claude-sonnet-5-5", "0.000003", "0.000015", FULL),
    entry("anthropic/claude-sonnet-5-5:batch", "0.0000015", "0.0000075", FULL),
    entry("anthropic/claude-old", "0.000001", "0.000005", ["tools", "temperature"]),
    entry("deepseek/deepseek-chat", "0.0000003", "0.0000012", FULL),
    entry("qwen/qwen3-coder:free", "0", "0", FULL),
    entry("meta/llama-free-plain", "0", "0", ["temperature"]),
    entry("mistral/no-structure", "0.000002", "0.000006", ["temperature"]),
    {"id": "sparse/model"},
]}


@pytest.fixture
def catalog() -> Catalog:
    return parse_catalog(PAYLOAD)


def test_prices_become_usd_per_million(catalog: Catalog) -> None:
    m = catalog.models["anthropic/claude-sonnet-5-5"]
    assert m.prompt_usd_per_m == pytest.approx(3.0)
    assert m.completion_usd_per_m == pytest.approx(15.0)
    assert m.context_length == 200000


def test_free_flag(catalog: Catalog) -> None:
    assert catalog.models["qwen/qwen3-coder:free"].free
    assert catalog.models["meta/llama-free-plain"].free
    assert not catalog.models["deepseek/deepseek-chat"].free


def test_structured_and_tools_properties(catalog: Catalog) -> None:
    assert catalog.models["deepseek/deepseek-chat"].structured_outputs
    assert catalog.models["deepseek/deepseek-chat"].tools
    old = catalog.models["anthropic/claude-old"]
    assert old.tools and not old.structured_outputs


def test_sparse_entry_is_tolerated(catalog: Catalog) -> None:
    m = catalog.models["sparse/model"]
    assert m.context_length == 0 and m.free and not m.structured_outputs


def test_get_matching_and_estimate(catalog: Catalog) -> None:
    assert catalog.get("nope") is None
    assert {m.id for m in catalog.matching(["deepseek/*", "qwen/qwen3-coder:free"])} == {
        "deepseek/deepseek-chat", "qwen/qwen3-coder:free"}
    assert catalog.estimate_usd("anthropic/claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(4.5)
    assert catalog.estimate_usd("nope", 1, 1) is None


def test_refuse_unstructured(catalog: Catalog) -> None:
    refuse_unstructured(catalog, ["deepseek/deepseek-chat", "qwen/qwen3-coder:free", AUTO])
    with pytest.raises(ModelRefused, match="unknown OpenRouter model"):
        refuse_unstructured(catalog, ["x/y"])
    with pytest.raises(ModelRefused, match="mistral/no-structure does not support structured outputs"):
        refuse_unstructured(catalog, ["mistral/no-structure"])


def test_vet_allow_list(catalog: Catalog) -> None:
    ids = {m.id for m in vet_allow_list(catalog, ["anthropic/*"])}
    assert ids == {"anthropic/claude-sonnet-5-5", "anthropic/claude-sonnet-5-5:batch"}
    ids = {m.id for m in vet_allow_list(catalog, ["anthropic/*"], ["*:batch"])}
    assert ids == {"anthropic/claude-sonnet-5-5"}
    with pytest.raises(ModelRefused, match="supports structured"):
        vet_allow_list(catalog, ["mistral/*"])
    with pytest.raises(ModelRefused, match="needs allowed_models"):
        vet_allow_list(catalog, [])


def counting_client(handler: Any) -> tuple[httpx2.AsyncClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def wrapped(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    return httpx2.AsyncClient(transport=httpx2.MockTransport(wrapped)), seen


def ok(_: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=PAYLOAD)


async def test_load_catalog_fetches_and_caches_without_the_key(tmp_path: Path) -> None:
    cache = tmp_path / "sub" / "models.json"
    client, seen = counting_client(ok)
    cat = await load_catalog(api_key="sk-or-secret", cache_path=cache, client=client)
    assert "deepseek/deepseek-chat" in cat.models
    assert seen[0].headers["authorization"] == "Bearer sk-or-secret"
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/models"
    assert "sk-or-secret" not in cache.read_text()
    assert json.loads(cache.read_text()) == PAYLOAD


async def test_load_catalog_without_key_sends_no_authorization(tmp_path: Path) -> None:
    client, seen = counting_client(ok)
    await load_catalog(api_key=None, cache_path=tmp_path / "m.json", client=client)
    assert "authorization" not in seen[0].headers


async def test_fresh_cache_makes_no_request(tmp_path: Path) -> None:
    cache = tmp_path / "m.json"
    cache.write_text(json.dumps(PAYLOAD))
    os.utime(cache, (1000, 1000))
    client, seen = counting_client(ok)
    cat = await load_catalog(api_key=None, cache_path=cache, client=client, now=lambda: 1000 + 3600)
    assert seen == [] and len(cat.models) == len(PAYLOAD["data"])


async def test_stale_cache_is_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "m.json"
    cache.write_text(json.dumps({"data": []}))
    os.utime(cache, (1000, 1000))
    client, seen = counting_client(ok)
    cat = await load_catalog(api_key=None, cache_path=cache, client=client, now=lambda: 1000 + 86401)
    assert len(seen) == 1 and "deepseek/deepseek-chat" in cat.models


async def test_failed_fetch_falls_back_to_stale_cache(tmp_path: Path) -> None:
    cache = tmp_path / "m.json"
    cache.write_text(json.dumps(PAYLOAD))
    os.utime(cache, (1000, 1000))
    client, seen = counting_client(lambda _: httpx2.Response(500))
    cat = await load_catalog(api_key=None, cache_path=cache, client=client, now=lambda: 10**7)
    assert len(seen) == 1 and "deepseek/deepseek-chat" in cat.models


async def test_failed_fetch_with_no_cache_raises(tmp_path: Path) -> None:
    client, _ = counting_client(lambda _: httpx2.Response(500))
    with pytest.raises(OpenRouterError):
        await load_catalog(api_key=None, cache_path=tmp_path / "m.json", client=client)


async def test_key_status() -> None:
    body = {"data": {"limit": 5, "limit_remaining": 4.5, "usage": 0.5,
                     "free_model_daily_requests": {"used": 0, "limit": 1000, "remaining": 990}}}
    client, seen = counting_client(lambda _: httpx2.Response(200, json=body))
    status = await key_status(api_key="sk-or-k", client=client)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/key"
    assert (status.limit, status.limit_remaining, status.usage) == (5.0, 4.5, 0.5)
    assert (status.free_daily_limit, status.free_daily_remaining) == (1000, 990)


async def test_key_status_without_limits() -> None:
    client, _ = counting_client(lambda _: httpx2.Response(200, json={"data": {"limit": None, "usage": 0}}))
    status = await key_status(api_key="k", client=client)
    assert status.limit is None and status.free_daily_limit is None


def test_render_models_default_hides_unstructured(catalog: Catalog) -> None:
    out = render_models(catalog)
    assert "mistral/no-structure" not in out and "anthropic/claude-old" not in out
    assert "deepseek/deepseek-chat" in out
    lines = out.splitlines()
    assert lines[0].startswith("id")
    ids = [ln.split()[0] for ln in lines[1:]]
    assert ids == sorted(ids, key=lambda i: (catalog.models[i].prompt_usd_per_m, i))
    assert "mistral/no-structure" in render_models(catalog, structured_only=False)


def test_render_models_filters(catalog: Catalog) -> None:
    free = render_models(catalog, free=True)
    assert "qwen/qwen3-coder:free" in free and "deepseek" not in free and "meta/llama-free-plain" not in free
    cheap = render_models(catalog, max_input_usd_per_m=1.0)
    assert "deepseek/deepseek-chat" in cheap and "claude-sonnet" not in cheap
    only = render_models(catalog, pattern="anthropic/*")
    assert "deepseek" not in only and "claude-sonnet-5-5" in only
    assert "\n" not in render_models(catalog, pattern="nothing/*")


def test_render_models_notes_column(catalog: Catalog) -> None:
    notes = {"deepseek/deepseek-chat": {"index": 61, "source": "artificialanalysis.ai", "as_of": "2026-10"}}
    out = render_models(catalog, notes=notes)
    assert "61 (artificialanalysis.ai, 2026-10)" in out


def test_render_models_json_round_trips(catalog: Catalog) -> None:
    rows = json.loads(render_models(catalog, as_json=True, free=True))
    assert [r["id"] for r in rows] == ["qwen/qwen3-coder:free"]
    assert rows[0]["free"] is True and rows[0]["structured_outputs"] is True
    assert json.loads(render_models(catalog, as_json=True, pattern="nothing/*")) == []


def test_load_notes(tmp_path: Path) -> None:
    path = tmp_path / "model-notes.toml"
    path.write_text(
        '"deepseek/deepseek-chat" = { index = 61, source = "aa", as_of = "2026-09" }\n\n'
        '["qwen/qwen3-coder"]\nindex = 55\nsource = "aa"\nas_of = "2026-10"\n'
    )
    notes = load_notes(path)
    assert notes["qwen/qwen3-coder"]["index"] == 55
    assert notes["deepseek/deepseek-chat"]["as_of"] == "2026-09"


def test_load_notes_missing_file(tmp_path: Path) -> None:
    assert load_notes(tmp_path / "nope.toml") == {}


def test_a_router_with_a_variable_price_is_listed_last_and_never_passes_a_price_filter() -> None:
    catalog = parse_catalog({"data": [
        entry("openrouter/auto", "-1", "-1", FULL), entry("a/cheap", "0.0000001", "0.0000002", FULL),
    ]})
    assert catalog.estimate_usd("openrouter/auto", 1000, 1000) is None
    ids = [r["id"] for r in json.loads(render_models(catalog, as_json=True))]
    assert ids == ["a/cheap", "openrouter/auto"]
    assert "varies" in render_models(catalog)
    assert [r["id"] for r in json.loads(render_models(catalog, max_input_usd_per_m=5.0, as_json=True))] == ["a/cheap"]


def test_a_routing_shortcut_suffix_is_looked_up_as_its_model() -> None:
    catalog = parse_catalog(PAYLOAD)
    refuse_unstructured(catalog, ["anthropic/claude-sonnet-5-5:nitro", "anthropic/claude-sonnet-5-5:floor"])
    with pytest.raises(ModelRefused, match="structured outputs"):
        refuse_unstructured(catalog, ["anthropic/claude-old:nitro"])
    with pytest.raises(ModelRefused, match="unknown"):
        refuse_unstructured(catalog, ["no/such-model:nitro"])
