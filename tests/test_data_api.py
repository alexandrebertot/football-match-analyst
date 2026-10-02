import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from football_agent.data_api import CACHE_TTL_SECONDS, open_cache, read_cache, write_cache


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = open_cache(tmp_path / "cache" / "responses.sqlite")
    yield db
    db.close()


def test_read_cache_returns_data_within_ttl(cache: sqlite3.Connection) -> None:
    data = {"standings": [{"team": "Paris Saint-Germain FC", "points": 21}]}
    write_cache(cache, "/competitions/FL1/standings", data, now=0)

    assert read_cache(cache, "/competitions/FL1/standings", now=CACHE_TTL_SECONDS) == data


def test_read_cache_returns_none_for_unknown_key(cache: sqlite3.Connection) -> None:
    assert read_cache(cache, "/competitions/FL1/standings", now=0) is None


def test_read_cache_returns_none_when_expired(cache: sqlite3.Connection) -> None:
    write_cache(cache, "/competitions/FL1/standings", {"standings": []}, now=0)

    assert read_cache(cache, "/competitions/FL1/standings", now=CACHE_TTL_SECONDS + 1) is None


def test_write_cache_replaces_existing_entry(cache: sqlite3.Connection) -> None:
    write_cache(cache, "/competitions/FL1/standings", {"version": 1}, now=0)
    write_cache(cache, "/competitions/FL1/standings", {"version": 2}, now=100)

    assert read_cache(cache, "/competitions/FL1/standings", now=100) == {"version": 2}
