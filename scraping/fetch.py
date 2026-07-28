"""Загрузка страниц и извлечение основного текста через trafilatura."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import trafilatura

import config
from scraping.cache import Cache

log = logging.getLogger(__name__)

_cache = Cache(config.CACHE_DB)


@dataclass
class FetchedPage:
    url: str
    text: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def fetch_page(url: str) -> FetchedPage:
    """Скачивает страницу и извлекает основной текст. Кэширует успех и неудачу."""
    cache_key = f"page:{url}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return FetchedPage(url=url, text=cached["text"], error=cached["error"])

    page = _fetch_uncached(url)
    _cache.set(cache_key, {"text": page.text, "error": page.error})
    return page


def _fetch_uncached(url: str) -> FetchedPage:
    headers = {"User-Agent": config.USER_AGENT}
    try:
        resp = httpx.get(
            url, headers=headers, timeout=config.FETCH_TIMEOUT, follow_redirects=True
        )
        if resp.status_code != 200:
            return FetchedPage(url=url, error=f"HTTP {resp.status_code}")
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if not text or len(text.split()) < 20:
            return FetchedPage(url=url, error="empty or too short after extraction")
        return FetchedPage(url=url, text=text[: config.MAX_SOURCE_CHARS])
    except Exception as exc:  # noqa: BLE001 — таймауты, SSL, редиректы и т.п.
        log.warning("fetch failed for %s: %s", url, exc)
        return FetchedPage(url=url, error=str(exc)[:200])
