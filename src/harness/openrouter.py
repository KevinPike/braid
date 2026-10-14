"""The OpenRouter model catalog and key status (specs/openrouter.md).

The catalog is ``GET /api/v1/models``, cached on disk for a day. It answers three questions without a model call:
does this model support structured outputs (Workflows need JSON), what does it cost per million tokens (the fallback
when a reply carries no ``cost``), and which models are worth listing. The cache holds the response, never the key.
"""

from __future__ import annotations

import fnmatch
import json
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2

BASE_URL = "https://openrouter.ai/api/v1"
AUTO = "openrouter/auto"


class OpenRouterError(RuntimeError):
    """A catalog or key request that failed, with no cache to fall back on."""


class ModelRefused(ValueError):
    """A model the harness will not call, and why."""


@dataclass(frozen=True)
class CatalogModel:
    id: str
    context_length: int
    prompt_usd_per_m: float
    completion_usd_per_m: float
    free: bool
    supported_parameters: frozenset[str]

    @property
    def variable(self) -> bool:
        """A router (``openrouter/auto``): the API reports its price as -1 because the served model sets it."""
        return self.prompt_usd_per_m < 0

    @property
    def structured_outputs(self) -> bool:
        return "structured_outputs" in self.supported_parameters

    @property
    def tools(self) -> bool:
        return "tools" in self.supported_parameters


@dataclass
class Catalog:
    models: dict[str, CatalogModel] = field(default_factory=dict)

    def get(self, model_id: str) -> CatalogModel | None:
        """The catalog entry for ``model_id``; a routing shortcut (``:nitro`` fastest, ``:floor`` cheapest) is not an entry
        of its own, so it is looked up as the model it modifies."""
        return self.models.get(model_id) or self.models.get(model_id.removesuffix(":nitro").removesuffix(":floor"))

    def matching(self, patterns: Sequence[str]) -> list[CatalogModel]:
        return [m for m in self.models.values() if any(fnmatch.fnmatchcase(m.id, p) for p in patterns)]

    def estimate_usd(self, model_id: str, input_tokens: int, output_tokens: int) -> float | None:
        m = self.models.get(model_id)
        if m is None or m.variable:
            return None
        return (input_tokens * m.prompt_usd_per_m + output_tokens * m.completion_usd_per_m) / 1e6


def refuse_unstructured(catalog: Catalog, model_ids: Iterable[str]) -> None:
    """Raise ``ModelRefused`` for a model that is not in the catalog or lacks structured outputs. ``AUTO`` passes."""
    for model_id in model_ids:
        if model_id == AUTO:
            continue
        m = catalog.get(model_id)
        if m is None:
            raise ModelRefused(f"unknown OpenRouter model {model_id!r}; `harness-workflow models` lists the catalog")
        if not m.structured_outputs:
            raise ModelRefused(f"{model_id} does not support structured outputs; Workflows need JSON")


def vet_allow_list(catalog: Catalog, allowed: Sequence[str], excluded: Sequence[str] = ()) -> list[CatalogModel]:
    """The structured-output models ``allowed`` names, less ``excluded``; ``ModelRefused`` when none are left."""
    if not allowed:
        raise ModelRefused(f"{AUTO} needs allowed_models")
    drop = {m.id for m in catalog.matching(excluded)} if excluded else set()
    kept = [m for m in catalog.matching(allowed) if m.structured_outputs and m.id not in drop]
    if not kept:
        raise ModelRefused(f"no model in allowed_models {list(allowed)} supports structured outputs")
    return kept


def _per_m(value: Any) -> float:
    try:
        return float(value) * 1e6
    except (TypeError, ValueError):
        return 0.0


def parse_catalog(payload: Mapping[str, Any]) -> Catalog:
    models: dict[str, CatalogModel] = {}
    for raw in payload.get("data") or []:
        model_id = raw.get("id")
        if not isinstance(model_id, str):
            continue
        pricing = raw.get("pricing") or {}
        prompt, completion = _per_m(pricing.get("prompt")), _per_m(pricing.get("completion"))
        models[model_id] = CatalogModel(
            id=model_id,
            context_length=int(raw.get("context_length") or 0),
            prompt_usd_per_m=prompt,
            completion_usd_per_m=completion,
            free=model_id.endswith(":free") or (prompt == 0 and completion == 0),
            supported_parameters=frozenset(raw.get("supported_parameters") or ()),
        )
    return Catalog(models)


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def _get(client: httpx2.AsyncClient | None, url: str, api_key: str | None) -> dict[str, Any]:
    try:
        if client is None:
            async with httpx2.AsyncClient(timeout=30.0) as own:
                resp = await own.get(url, headers=_headers(api_key))
        else:
            resp = await client.get(url, headers=_headers(api_key))
        resp.raise_for_status()
        body = resp.json()
    except (httpx2.HTTPError, ValueError) as exc:
        raise OpenRouterError(f"GET {url} failed: {type(exc).__name__}") from exc
    if not isinstance(body, dict):
        raise OpenRouterError(f"GET {url} returned {type(body).__name__}, not an object")
    return body


async def load_catalog(
    *, api_key: str | None, cache_path: Path, client: httpx2.AsyncClient | None = None, max_age_s: float = 86400.0,
    now: Callable[[], float] = time.time, base_url: str = BASE_URL,
) -> Catalog:
    """The catalog from a cache younger than ``max_age_s``, else fetched and cached; a stale cache covers a failed fetch."""
    cached: Catalog | None = None
    fresh = False
    try:
        cached = parse_catalog(json.loads(cache_path.read_text()))
        fresh = now() - cache_path.stat().st_mtime < max_age_s
    except (OSError, ValueError, AttributeError):
        cached = None
    if cached is not None and fresh:
        return cached
    try:
        body = await _get(client, f"{base_url}/models", api_key)
    except OpenRouterError:
        if cached is not None:
            return cached
        raise
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(body))
    tmp.replace(cache_path)  # a reader never sees half a catalog
    return parse_catalog(body)


@dataclass(frozen=True)
class KeyStatus:
    limit: float | None
    limit_remaining: float | None
    usage: float
    free_daily_limit: int | None
    free_daily_remaining: int | None


def _opt(value: Any, kind: type[float] | type[int]) -> Any:
    return None if value is None else kind(value)


async def key_status(*, api_key: str, client: httpx2.AsyncClient | None = None, base_url: str = BASE_URL) -> KeyStatus:
    data = (await _get(client, f"{base_url}/key", api_key)).get("data") or {}
    free = data.get("free_model_daily_requests") or {}
    return KeyStatus(
        limit=_opt(data.get("limit"), float), limit_remaining=_opt(data.get("limit_remaining"), float),
        usage=float(data.get("usage") or 0), free_daily_limit=_opt(free.get("limit"), int),
        free_daily_remaining=_opt(free.get("remaining"), int),
    )


def load_notes(path: Path) -> dict[str, dict[str, Any]]:
    """Hand-kept ranking notes (``index``, ``source``, ``as_of``) by model id; a missing file is no notes."""
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        return {}
    return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}


def _note(notes: Mapping[str, Mapping[str, Any]] | None, model_id: str) -> str:
    n = (notes or {}).get(model_id)
    if not n:
        return ""
    return f"{n.get('index', '?')} ({n.get('source', '?')}, {n.get('as_of', '?')})"


def render_models(
    catalog: Catalog, *, free: bool = False, structured_only: bool = True, max_input_usd_per_m: float | None = None,
    notes: Mapping[str, Mapping[str, Any]] | None = None, as_json: bool = False, pattern: str | None = None,
) -> str:
    """The catalog as a table (or a JSON list) sorted by input price then id, after the filters."""
    rows = [
        m for m in catalog.models.values()
        if (not free or m.free) and (not structured_only or m.structured_outputs)
        and (max_input_usd_per_m is None or (not m.variable and m.prompt_usd_per_m <= max_input_usd_per_m))
        and (pattern is None or fnmatch.fnmatchcase(m.id, pattern))
    ]
    rows.sort(key=lambda m: (m.variable, m.prompt_usd_per_m, m.id))  # routers (price set per call) last
    if as_json:
        return json.dumps([
            {
                "id": m.id, "context_length": m.context_length, "prompt_usd_per_m": m.prompt_usd_per_m,
                "completion_usd_per_m": m.completion_usd_per_m, "free": m.free, "tools": m.tools,
                "structured_outputs": m.structured_outputs, "note": notes.get(m.id) if notes else None,
            }
            for m in rows
        ])
    if not rows:
        return "no models match"
    head = ["id", "ctx", "in $/M", "out $/M", "free", "tools", "ranking"]
    body = [
        [m.id, str(m.context_length), "varies" if m.variable else f"{m.prompt_usd_per_m:.3f}",
         "varies" if m.variable else f"{m.completion_usd_per_m:.3f}",
         "yes" if m.free else "", "yes" if m.tools else "", _note(notes, m.id)]
        for m in rows
    ]
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in [head, *body])
