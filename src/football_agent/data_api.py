import json
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

BASE_URL = "https://api.football-data.org/v4"
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


def make_http_client(api_key: str) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers={"X-Auth-Token": api_key}, timeout=10.0)


class FootballDataClient:
    def __init__(self, http: httpx.Client, cache: sqlite3.Connection) -> None:
        self.http = http
        self.cache = cache

    def get_standings(self, competition: str) -> dict[str, Any]:
        return self._get(f"/competitions/{competition}/standings", {})

    def get_matches(self, competition: str, date_from: date, date_to: date) -> dict[str, Any]:
        params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}
        return self._get(f"/competitions/{competition}/matches", params)

    def get_team_matches(self, team_id: int, limit: int) -> dict[str, Any]:
        return self._get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": limit})

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        key = f"{path}?{urlencode(params)}"
        cached = read_cache(self.cache, key, now=time.time())
        if cached is not None:
            return cached
        response = self.http.get(path, params=params)
        response.raise_for_status()
        data = response.json()
        write_cache(self.cache, key, data, now=time.time())
        return data
