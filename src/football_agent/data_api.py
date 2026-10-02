import json
import sqlite3
from pathlib import Path
from typing import Any

CACHE_TTL_SECONDS = 3600


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS responses ("
        "key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at REAL NOT NULL)"
    )
    return db


def read_cache(db: sqlite3.Connection, key: str, now: float) -> dict[str, Any] | None:
    """Return the cached response for `key`, or None if it is missing or older than the TTL."""
    row = db.execute("SELECT body, fetched_at FROM responses WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    body, fetched_at = row
    if now - fetched_at > CACHE_TTL_SECONDS:
        return None
    return json.loads(body)


def write_cache(db: sqlite3.Connection, key: str, data: dict[str, Any], now: float) -> None:
    with db:
        db.execute(
            "INSERT OR REPLACE INTO responses (key, body, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(data), now),
        )
