import sqlite3
from datetime import date

import httpx
import pytest

from football_agent.data_api import (
    BASE_URL,
    CACHE_TTL_SECONDS,
    FootballDataClient,
    make_http_client,
    read_cache,
    write_cache,
)


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


def fake_client(
    cache: sqlite3.Connection, requests: list[httpx.Request], status_code: int = 200
) -> FootballDataClient:
    """Return a client whose HTTP calls hit a fake API that records every request it receives."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json={"path": request.url.path})

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))
    return FootballDataClient(http, cache)


def test_make_http_client_sets_base_url_and_api_key() -> None:
    http = make_http_client("test-key")

    assert http.base_url == BASE_URL + "/"
    assert http.headers["X-Auth-Token"] == "test-key"


def test_get_standings_requests_competition_standings(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    data = client.get_standings("FL1")

    assert requests[0].url.path == "/v4/competitions/FL1/standings"
    assert "season" not in requests[0].url.params
    assert data == {"path": "/v4/competitions/FL1/standings"}


def test_get_standings_sends_requested_season(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    client.get_standings("FL1", season=2025)

    assert requests[0].url.params["season"] == "2025"


def test_standings_of_different_seasons_are_cached_separately(
    cache: sqlite3.Connection,
) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    client.get_standings("FL1")
    client.get_standings("FL1", season=2025)

    assert len(requests) == 2


def test_get_teams_requests_competition_teams(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    client.get_teams("FL1")

    assert requests[0].url.path == "/v4/competitions/FL1/teams"


def test_get_matches_sends_date_range(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    client.get_matches("PL", date(2026, 10, 1), date(2026, 10, 7))

    assert requests[0].url.path == "/v4/competitions/PL/matches"
    assert requests[0].url.params["dateFrom"] == "2026-10-01"
    assert requests[0].url.params["dateTo"] == "2026-10-07"


def test_get_team_matches_requests_last_finished_matches_of_competition(
    cache: sqlite3.Connection,
) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    client.get_team_matches(524, "FL1", limit=5)

    assert requests[0].url.path == "/v4/teams/524/matches"
    assert requests[0].url.params["status"] == "FINISHED"
    assert requests[0].url.params["competitions"] == "FL1"
    assert requests[0].url.params["limit"] == "5"


def test_identical_call_is_served_from_cache(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests)

    first = client.get_standings("FL1")
    second = client.get_standings("FL1")

    assert len(requests) == 1
    assert second == first


def test_http_error_is_raised_and_not_cached(cache: sqlite3.Connection) -> None:
    requests: list[httpx.Request] = []
    client = fake_client(cache, requests, status_code=429)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            client.get_standings("FL1")

    assert len(requests) == 2
