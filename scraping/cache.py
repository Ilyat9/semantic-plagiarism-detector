"""sqlite-кэш для поисковой выдачи и скачанных страниц."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # неделя — для успешных результатов
ERROR_TTL_SECONDS = 30 * 60  # 30 минут — для ошибок (таймауты не «замораживают» источник)


class Cache:
    """Потокобезопасный kv-кэш с TTL на запись.

    Одно соединение на инстанс с check_same_thread=False + Lock: эндпоинт /check
    работает в threadpool, а sqlite3 по умолчанию запрещает шаринг соединения
    между потоками.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)"
            )
            self._conn.commit()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT value, ts FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            value, ts = row
            entry = json.loads(value)
            # Записи старого формата (без ttl) считаем протухшими
            ttl = entry.get("ttl")
            if ttl is None or time.time() - ts > ttl:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            return entry["value"]

    def set(self, key: str, value, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        entry = {"value": value, "ttl": ttl}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
                (key, json.dumps(entry, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
