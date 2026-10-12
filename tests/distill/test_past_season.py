from datetime import date
from typing import Any

import httpx
import pytest

from football_agent.agent_eval.snapshot import SnapshotClient
from football_agent.data_api import BASE_URL
from football_agent.distill.past_season import (
    RESOURCES,
    download_season,
    snapshot_as_of,
    standings_table,
)
from football_agent.tools import COMPETITIONS, ToolContext, get_matches, get_standings

ARSENAL, CHELSEA, EVERTON = 57, 61, 62
TEAMS = {
    ARSENAL: ("Arsenal FC", "Arsenal"),
    CHELSEA: ("Chelsea FC", "Chelsea"),
    EVERTON: ("Everton FC", "Everton"),
}


def team(team_id: int) -> dict[str, Any]:
    name, short_name = TEAMS[team_id]
    return {"id": team_id, "name": name, "shortName": short_name, "tla": short_name[:3].upper()}


def match(
    day: str, home: int, away: int, goals: tuple[int, int], stage: str = "REGULAR_SEASON"
) -> dict[str, Any]:
    home_goals, away_goals = goals
    winner = "DRAW" if home_goals == away_goals else "HOME_TEAM"
    if home_goals < away_goals:
        winner = "AWAY_TEAM"
    return {
        "utcDate": f"{day}T15:00:00Z",
        "status": "FINISHED",
        "matchday": 1,
        "stage": stage,
        "competition": {"code": "PL", "name": "Premier League"},
        "season": {"startDate": "2025-08-15", "endDate": "2026-05-24"},
        "homeTeam": team(home),
        "awayTeam": team(away),
        "score": {
            "winner": winner,
            "duration": "REGULAR",
            "fullTime": {"home": home_goals, "away": away_goals},
            "halfTime": {"home": 0, "away": 0},
        },
    }


SEASON_MATCHES = [
    match("2025-08-16", ARSENAL, CHELSEA, (2, 0)),
    match("2025-08-23", CHELSEA, EVERTON, (1, 1)),
    match("2025-08-30", EVERTON, ARSENAL, (0, 3)),
]
SEASON = {
    "season": 2025,
    "competitions": {
        "PL": {
            "matches": {"competition": {"name": "Premier League"}, "matches": SEASON_MATCHES},
            "teams": {"teams": [team(team_id) for team_id in TEAMS]},
        }
    },
}


def points(table: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return [(row["team"]["shortName"], row["points"]) for row in table]


def test_standings_table_counts_results_and_ranks_on_points_then_goals() -> None:
    teams = [team(team_id) for team_id in TEAMS]

    table = standings_table(teams, SEASON_MATCHES)

    # Chelsea and Everton have one point each: Chelsea's goal difference (-2) beats Everton's (-3).
    assert points(table) == [("Arsenal", 6), ("Chelsea", 1), ("Everton", 1)]
    arsenal = {key: value for key, value in table[0].items() if key != "team"}
    assert arsenal == {
        "position": 1,
        "playedGames": 2,
        "won": 2,
        "draw": 0,
        "lost": 0,
        "goalsFor": 5,
        "goalsAgainst": 0,
        "goalDifference": 5,
        "points": 6,
    }


def test_standings_table_breaks_a_complete_tie_on_the_name() -> None:
    teams = [team(EVERTON), team(CHELSEA)]

    assert points(standings_table(teams, [])) == [("Chelsea", 0), ("Everton", 0)]


def test_snapshot_as_of_replays_the_season_on_the_morning_of_a_day() -> None:
    snapshot = snapshot_as_of(SEASON, date(2025, 8, 23))
    recorded = snapshot["competitions"]["PL"]

    table = recorded["standings"]["standings"][0]["table"]
    assert snapshot["date"] == "2025-08-23"
    assert points(table) == [("Arsenal", 3), ("Everton", 0), ("Chelsea", 0)]
    statuses = [
        (m["utcDate"][:10], m["status"], m["score"]["fullTime"])
        for m in recorded["matches"]["matches"]
    ]
    assert statuses == [
        ("2025-08-16", "FINISHED", {"home": 2, "away": 0}),
        ("2025-08-23", "TIMED", {"home": None, "away": None}),
        ("2025-08-30", "TIMED", {"home": None, "away": None}),
    ]


def test_snapshot_as_of_before_the_first_match_has_every_team_on_zero() -> None:
    snapshot = snapshot_as_of(SEASON, date(2025, 8, 1))

    table = snapshot["competitions"]["PL"]["standings"]["standings"][0]["table"]
    assert points(table) == [("Arsenal", 0), ("Chelsea", 0), ("Everton", 0)]


def test_snapshot_as_of_counts_an_awarded_result_and_keeps_it_as_recorded() -> None:
    awarded = match("2025-08-23", CHELSEA, EVERTON, (0, 0)) | {"status": "AWARDED"}
    season = {
        "season": 2025,
        "competitions": {
            "PL": {
                "matches": {"competition": {"name": "Premier League"}, "matches": [awarded]},
                "teams": {"teams": [team(team_id) for team_id in TEAMS]},
            }
        },
    }

    recorded = snapshot_as_of(season, date(2025, 9, 1))["competitions"]["PL"]

    table = recorded["standings"]["standings"][0]["table"]
    assert points(table) == [("Chelsea", 1), ("Everton", 1), ("Arsenal", 0)]
    assert recorded["matches"]["matches"] == [awarded]


def test_snapshot_as_of_leaves_knockout_matches_out_of_the_table() -> None:
    knockout = match("2026-03-10", CHELSEA, EVERTON, (4, 0), stage="LAST_16")
    season = {
        "season": 2025,
        "competitions": {
            "CL": {
                "matches": {
                    "competition": {"name": "UEFA Champions League"},
                    "matches": [
                        match("2025-09-16", ARSENAL, CHELSEA, (1, 0), "LEAGUE_STAGE"),
                        knockout,
                    ],
                },
                "teams": {"teams": [team(team_id) for team_id in TEAMS]},
            }
        },
    }

    snapshot = snapshot_as_of(season, date(2026, 5, 1))

    table = snapshot["competitions"]["CL"]["standings"]["standings"][0]["table"]
    assert points(table) == [("Arsenal", 3), ("Everton", 0), ("Chelsea", 0)]


def test_snapshot_as_of_is_served_to_the_tools_like_an_evaluation_snapshot() -> None:
    day = date(2025, 8, 23)
    context = ToolContext(SnapshotClient(snapshot_as_of(SEASON, day)), None, day)

    standings = get_standings(context, "PL")
    upcoming = get_matches(context, "PL", team_name="Everton")

    assert (standings["season"], standings["table"][0]["team"]) == ("2025-26", "Arsenal")
    assert [(m["home"], m["away"], m["status"]) for m in upcoming] == [
        ("Chelsea", "Everton", "TIMED"),
        ("Everton", "Arsenal", "TIMED"),
    ]


def test_download_season_downloads_every_resource_of_every_competition() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"path": request.url.path})

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))

    season = download_season(http, 2025, pause_seconds=0)

    assert len(requests) == len(COMPETITIONS) * len(RESOURCES)
    assert {request.url.params["season"] for request in requests} == {"2025"}
    assert season["competitions"]["PL"]["teams"] == {"path": "/v4/competitions/PL/teams"}


def test_download_season_stops_when_the_api_refuses_a_request() -> None:
    refusing = httpx.MockTransport(lambda request: httpx.Response(429))
    http = httpx.Client(base_url=BASE_URL, transport=refusing)

    with pytest.raises(httpx.HTTPStatusError):
        download_season(http, 2025, pause_seconds=0)
