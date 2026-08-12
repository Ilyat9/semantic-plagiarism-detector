"""Загрузка страниц и извлечение основного текста через trafilatura."""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

import config
from scraping.cache import DEFAULT_TTL_SECONDS, ERROR_TTL_SECONDS, Cache

log = logging.getLogger(__name__)

_cache = Cache(config.CACHE_DB)

_MAX_REDIRECTS = 3


@dataclass
class FetchedPage:
    url: str
    text: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def fetch_page(url: str) -> FetchedPage:
    """Скачивает страницу и извлекает основной текст.

    Успех кэшируется на неделю, ошибка — на ERROR_TTL_SECONDS (разовый таймаут
    не должен надолго «замораживать» источник и резать recall).
    """
    cache_key = f"page:{url}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return FetchedPage(url=url, text=cached["text"], error=cached["error"])

    page = _fetch_uncached(url)
    _cache.set(
        cache_key,
        {"text": page.text, "error": page.error},
        ttl=ERROR_TTL_SECONDS if page.error else DEFAULT_TTL_SECONDS,
    )
    return page


def _is_safe_url(url: str) -> bool:
    """SSRF-фильтр: только http/https и только публичные адреса.

    URL приходят из поисковой выдачи по фразам пользовательского документа;
    подконтрольная страница могла бы увести запрос на metadata-сервис
    (169.254.169.254) или внутренний хост — отклоняем loopback/private/link-local.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        infos = socket.getaddrinfo(parsed.hostname, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False
        return True
    except (OSError, ValueError):
        return False


def _fetch_uncached(url: str) -> FetchedPage:
    if not _is_safe_url(url):
        return FetchedPage(url=url, error="url rejected (scheme or private address)")
    headers = {"User-Agent": config.USER_AGENT}
    try:
        with httpx.Client(headers=headers, timeout=config.FETCH_TIMEOUT) as client:
            # Редиректы обрабатываем вручную: каждый следующий URL тоже валидируем
            for _ in range(_MAX_REDIRECTS + 1):
                resp = client.get(url, follow_redirects=False)
                if resp.is_redirect:
                    url = urljoin(url, resp.headers["location"])
                    if not _is_safe_url(url):
                        return FetchedPage(url=url, error="redirect rejected (private address)")
                    continue
                break
            else:
                return FetchedPage(url=url, error="too many redirects")
        if resp.status_code != 200:
            return FetchedPage(url=url, error=f"HTTP {resp.status_code}")
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if not text or len(text.split()) < 20:
            return FetchedPage(url=url, error="empty or too short after extraction")
        return FetchedPage(url=url, text=text[: config.MAX_SOURCE_CHARS])
    except Exception as exc:  # noqa: BLE001 — таймауты, SSL, редиректы и т.п.
        log.warning("fetch failed for %s: %s", url, exc)
        return FetchedPage(url=url, error=str(exc)[:200])
