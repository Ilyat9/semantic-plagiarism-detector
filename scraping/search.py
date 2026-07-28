"""Веб-поиск: ddgs (основной) и SearXNG (fallback), с кэшированием выдачи."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

import config
from scraping.cache import Cache

log = logging.getLogger(__name__)

_cache = Cache(config.CACHE_DB)


@dataclass
class SearchResult:
    phrase: str
    urls: list[str] = field(default_factory=list)
    backend: str = ""
    error: str | None = None


def _search_ddgs(phrase: str, max_results: int) -> list[str]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = ddgs.text(f'"{phrase}"', max_results=max_results)
    return [r["href"] for r in results if r.get("href")]


def _search_searxng(phrase: str, max_results: int) -> list[str]:
    resp = httpx.get(
        f"{config.SEARXNG_URL}/search",
        params={"q": f'"{phrase}"', "format": "json"},
        timeout=config.FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [r["url"] for r in data.get("results", [])[:max_results] if r.get("url")]


_BACKENDS = {"ddgs": _search_ddgs, "searxng": _search_searxng}


def search_phrase(phrase: str, max_results: int | None = None) -> SearchResult:
    """Ищет точную фразу, возвращает топ-URL. Результаты кэшируются."""
    max_results = max_results or config.TOP_K_URLS
    cache_key = f"search:{phrase}:{max_results}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return SearchResult(
            phrase=phrase, urls=cached["urls"], backend=cached["backend"] + " (cache)"
        )

    backend = config.SEARCH_BACKEND
    order = ["ddgs", "searxng"] if backend == "auto" else [backend]
    last_error: Exception | None = None
    for name in order:
        try:
            urls = _BACKENDS[name](phrase, max_results)
            _cache.set(cache_key, {"urls": urls, "backend": name})
            return SearchResult(phrase=phrase, urls=urls, backend=name)
        except Exception as exc:  # noqa: BLE001 — сетевые ошибки разнообразны
            log.warning("search backend %s failed for %r: %s", name, phrase, exc)
            last_error = exc
    return SearchResult(phrase=phrase, error=str(last_error))


def search_phrases(phrases: list[str], max_results: int | None = None) -> list[SearchResult]:
    return [search_phrase(p, max_results) for p in phrases]
