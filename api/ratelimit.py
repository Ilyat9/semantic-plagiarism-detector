"""Rate limiting middleware для FastAPI.

Использует in-memory sliding window (подходит для single-instance; состояние
не переживает рестарт и не шарится между репликами — для multi-instance нужен Redis).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import config


@dataclass
class _Window:
    requests: list[float] = field(default_factory=list)


# IP -> endpoint prefix -> window
_store: dict[str, dict[str, _Window]] = {}
_last_sweep: float = 0.0
_SWEEP_INTERVAL = 300.0  # сек; вытеснение пустых окон и осиротевших IP

# Лимиты: prefix пути -> (max_requests, window_seconds)
LIMITS: dict[str, tuple[int, int]] = {
    "/check": (5, 60),  # 5 проверок в минуту
    "/report/": (20, 60),  # 20 PDF-скачиваний в минуту
}
# Дефолтный лимит для всех остальных путей
DEFAULT_LIMIT: tuple[int, int] = (60, 60)


def _clean_old(window: _Window, cutoff: float) -> None:
    window.requests = [t for t in window.requests if t > cutoff]


def _sweep(now: float) -> None:
    """Удаляет пустые окна и IP без окон, чтобы _store не рос бесконечно."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL:
        return
    _last_sweep = now
    max_window = max((w for _, w in list(LIMITS.values()) + [DEFAULT_LIMIT]))
    cutoff = now - max_window
    for ip in list(_store):
        windows = _store[ip]
        for prefix in list(windows):
            _clean_old(windows[prefix], cutoff)
            if not windows[prefix].requests:
                del windows[prefix]
        if not windows:
            del _store[ip]


def _match_limit(path: str) -> tuple[str, int, int]:
    for prefix, (max_req, window_sec) in LIMITS.items():
        if path.startswith(prefix):
            return prefix, max_req, window_sec
    return "*", *DEFAULT_LIMIT


def _is_allowed(client_ip: str, path: str, now: float) -> tuple[bool, int]:
    prefix, max_req, window_sec = _match_limit(path)
    window = _store.setdefault(client_ip, {}).setdefault(prefix, _Window())
    _clean_old(window, now - window_sec)
    if len(window.requests) >= max_req:
        if window.requests:
            retry_after = max(1, int(window_sec - (now - window.requests[0])))
        else:
            retry_after = window_sec
        return False, retry_after
    window.requests.append(now)
    return True, 0


def _client_ip(request: Request) -> str:
    if config.TRUST_X_FORWARDED_FOR:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        now = time.time()
        _sweep(now)
        allowed, retry_after = _is_allowed(_client_ip(request), request.url.path, now)
        if not allowed:
            # NB: raise HTTPException из BaseHTTPMiddleware доходит до
            # ServerErrorMiddleware и превращается в 500 — поэтому возвращаем ответ.
            return JSONResponse(
                status_code=429,
                content={"detail": "Слишком много запросов. Попробуйте позже."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
