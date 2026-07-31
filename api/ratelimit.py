"""Rate limiting middleware для FastAPI.

Использует in-memory sliding window (подходит для single-instance).
Для production с multiple replicas — заменить на Redis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class _Window:
    requests: list[float] = field(default_factory=list)


# IP -> endpoint -> window
_store: dict[str, dict[str, _Window]] = {}

# Лимиты: endpoint -> (max_requests, window_seconds)
LIMITS: dict[str, tuple[int, int]] = {
    "/check": (5, 60),  # 5 запросов в минуту
    "/report/": (20, 60),  # 20 PDF-скачиваний в минуту
}


def _clean_old(window: _Window, cutoff: float) -> None:
    window.requests = [t for t in window.requests if t > cutoff]


def _is_allowed(client_ip: str, endpoint: str, now: float) -> bool:
    for prefix, (max_req, window_sec) in LIMITS.items():
        if endpoint.startswith(prefix):
            if client_ip not in _store:
                _store[client_ip] = {}
            if prefix not in _store[client_ip]:
                _store[client_ip][prefix] = _Window()
            w = _store[client_ip][prefix]
            _clean_old(w, now - window_sec)
            if len(w.requests) >= max_req:
                return False
            w.requests.append(now)
            return True
    return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        if not _is_allowed(client_ip, request.url.path, now):
            raise HTTPException(
                status_code=429,
                detail="Слишком много запросов. Попробуйте позже.",
            )
        return await call_next(request)
