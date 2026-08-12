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

# Верхний лимит чанков на источник в stage 1 (скорость). Источники длиннее
# лимита молча усекаются — это компромисс recall/rate (см. README).
MAX_SOURCE_CHUNKS = int(os.getenv("PLAGCHECK_MAX_SOURCE_CHUNKS", "150"))

# Максимальный размер загружаемого документа, байт (по умолчанию 10 МБ)
MAX_UPLOAD_BYTES = int(os.getenv("PLAGCHECK_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Доверять заголовку X-Forwarded-For при определении IP клиента для rate limit.
# Включать только за доверенным реверс-прокси, иначе лимит обходится подменой заголовка.
TRUST_X_FORWARDED_FOR = os.getenv("PLAGCHECK_TRUST_X_FORWARDED_FOR", "").lower() in ("1", "true")

DATA_DIR = Path(os.getenv("PLAGCHECK_DATA_DIR", "data"))
CACHE_DB = DATA_DIR / "cache.sqlite"

# Идентифицирующий UA: подмена под браузер — это обход ограничений, а не «уважение».
# Часть сайтов может отвечать 403 — такие источники фиксируются в отчёте как недоступные.
USER_AGENT = os.getenv(
    "PLAGCHECK_USER_AGENT",
    "plagcheck/0.1 (+https://github.com/ilyat9/semantic-plagiarism-detector)",
)
