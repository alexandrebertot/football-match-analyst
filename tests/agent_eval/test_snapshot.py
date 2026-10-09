from datetime import date
from typing import Any

import httpx
import pytest

from football_agent.agent_eval.snapshot import RESOURCES, SnapshotClient, record_snapshot
from football_agent.data_api import BASE_URL
from football_agent.tools import COMPETITIONS

PSG, LYON, NICE = 524, 523, 522


def match(day: str, home: int, away: int, status: str = "FINISHED") -> dict[str, Any]:
    return {
        "utcDate": f"{day}T19:00:00Z",
        "status": status,
        "homeTeam": {"id": home},
        "awayTeam": {"id": away},
    }


SEASON_MATCHES = [
    match("2026-08-16", PSG, LYON),
    match("2026-08-23", NICE, PSG),
    match("2026-08-30", LYON, NICE),
    match("2026-09-13", PSG, NICE),
    match("2026-10-17", LYON, PSG, status="TIMED"),
]
SNAPSHOT = {
    "date": "2026-10-09",
    "competitions": {
        "FL1": {
            "standings": {"season": {"startDate": "2026-08-14"}, "standings": []},
            "teams": {"teams": [{"id": PSG}, {"id": LYON}, {"id": NICE}]},
            # Recorded in an order the client must not rely on.
            "matches": {"matches": list(reversed(SEASON_MATCHES))},
        }
    },
}


@pytest.fixture
def client() -> SnapshotClient:
    return SnapshotClient(SNAPSHOT)


def dates(raw: dict[str, Any]) -> list[str]:
    return [match["utcDate"][:10] for match in raw["matches"]]


def test_get_matches_keeps_the_matches_between_both_dates_included(client: SnapshotClient) -> None:
    raw = client.get_matches("FL1", date(2026, 8, 23), date(2026, 9, 13))

    assert sorted(dates(raw)) == ["2026-08-23", "2026-08-30", "2026-09-13"]


def test_get_team_matches_gives_the_last_finished_matches_oldest_first(
    client: SnapshotClient,
) -> None:
    raw = client.get_team_matches(PSG, "FL1", limit=2)

    assert dates(raw) == ["2026-08-23", "2026-09-13"]


def test_get_team_matches_with_a_zero_limit_gives_no_match(client: SnapshotClient) -> None:
    assert client.get_team_matches(PSG, "FL1", limit=0) == {"matches": []}


def test_get_standings_serves_the_recorded_season_only(client: SnapshotClient) -> None:
    assert client.get_standings("FL1") == client.get_standings("FL1", season=2026)

    with pytest.raises(ValueError, match="season starting in 2026"):
        client.get_standings("FL1", season=2025)


def test_get_teams_returns_the_recorded_teams(client: SnapshotClient) -> None:
    assert client.get_teams("FL1") == SNAPSHOT["competitions"]["FL1"]["teams"]


def test_unknown_competition_is_an_error_the_agent_can_act_on(client: SnapshotClient) -> None:
    with pytest.raises(ValueError, match="Unknown competition 'L1'"):
        client.get_teams("L1")


def test_record_snapshot_downloads_every_resource_of_every_competition() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"path": request.url.path})

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))

    snapshot = record_snapshot(http, date(2026, 10, 9), pause_seconds=0)

    assert len(requests) == len(COMPETITIONS) * len(RESOURCES)
    assert snapshot["date"] == "2026-10-09"
    assert list(snapshot["competitions"]) == COMPETITIONS
    assert snapshot["competitions"]["PL"]["teams"] == {"path": "/v4/competitions/PL/teams"}


def test_record_snapshot_stops_when_the_api_refuses_a_request() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))

    with pytest.raises(httpx.HTTPStatusError):
        record_snapshot(http, date(2026, 10, 9), pause_seconds=0)
