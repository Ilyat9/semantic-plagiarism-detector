"""Конфигурация пайплайна. Значения можно переопределить переменными окружения."""

from __future__ import annotations

import os
from pathlib import Path

# search_backend: ddgs | searxng | auto (ddgs с fallback на searxng)
SEARCH_BACKEND = os.getenv("PLAGCHECK_SEARCH_BACKEND", "auto")
SEARXNG_URL = os.getenv("PLAGCHECK_SEARXNG_URL", "http://localhost:8080")

TOP_K_URLS = int(os.getenv("PLAGCHECK_TOP_K_URLS", "5"))
FETCH_TIMEOUT = float(os.getenv("PLAGCHECK_FETCH_TIMEOUT", "15"))
MAX_SOURCE_CHARS = int(os.getenv("PLAGCHECK_MAX_SOURCE_CHARS", "20000"))

DATA_DIR = Path(os.getenv("PLAGCHECK_DATA_DIR", "data"))
CACHE_DB = DATA_DIR / "cache.sqlite"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
