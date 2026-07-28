"""sqlite-кэш для поисковой выдачи и скачанных страниц."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_TTL_SECONDS = 7 * 24 * 3600  # неделя


class Cache:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)"
        )
        self._conn.commit()

    def get(self, key: str):
        row = self._conn.execute("SELECT value, ts FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        value, ts = row
        if time.time() - ts > _TTL_SECONDS:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return json.loads(value)

    def set(self, key: str, value) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
