from typing import Any

import pytest

from football_agent.tools import (
    find_team,
    format_score,
    match_result,
    normalize_name,
    summarize_match,
    summarize_matches,
    summarize_standings,
    summarize_team_form,
)

TEAMS = [
    {"id": 524, "name": "Paris Saint-Germain FC", "shortName": "PSG"},
    {"id": 1045, "name": "Paris FC", "shortName": "Paris FC"},
    {"id": 523, "name": "Olympique Lyonnais", "shortName": "Olympique Lyon"},
    {"id": 516, "name": "Olympique de Marseille", "shortName": "Marseille"},
    {"id": 527, "name": "AS Saint-Étienne", "shortName": "Saint-Étienne"},
    {"id": 98, "name": "AC Milan", "shortName": "Milan"},
    {"id": 108, "name": "FC Internazionale Milano", "shortName": "Inter"},
]


def make_match(
    home: str, away: str, full_time: dict[str, int | None], status: str
) -> dict[str, Any]:
    """Return a match shaped like the football-data.org API, with only the fields we read."""
    return {
        "utcDate": "2026-09-19T15:15:00Z",
        "status": status,
        "matchday": 5,
        "competition": {"code": "FL1", "name": "Ligue 1"},
        "homeTeam": {"id": 1045, "name": f"{home} full name", "shortName": home},
        "awayTeam": {"id": 576, "name": f"{away} full name", "shortName": away},
        "score": {
            "winner": None,
            "duration": "REGULAR",
            "fullTime": full_time,
            "halfTime": {"home": None, "away": None},
        },
    }


def test_format_score_joins_home_and_away_goals() -> None:
    assert format_score({"home": 2, "away": 1}) == "2-1"


def test_format_score_returns_none_for_unplayed_match() -> None:
    assert format_score({"home": None, "away": None}) is None


def test_summarize_standings_keeps_only_useful_fields() -> None:
    raw = {
        "standings": [
            {
                "type": "TOTAL",
                "table": [
                    {
                        "position": 1,
                        "team": {"id": 548, "name": "AS Monaco FC", "shortName": "Monaco"},
                        "playedGames": 5,
                        "form": None,
                        "won": 4,
                        "draw": 1,
                        "lost": 0,
                        "points": 13,
                        "goalsFor": 8,
                        "goalsAgainst": 3,
                        "goalDifference": 5,
                    }
                ],
            }
        ]
    }

    assert summarize_standings(raw) == [
        {
            "position": 1,
            "team": "Monaco",
            "played": 5,
            "won": 4,
            "draw": 1,
            "lost": 0,
            "goals_for": 8,
            "goals_against": 3,
            "goal_difference": 5,
            "points": 13,
        }
    ]


def test_summarize_match_keeps_only_useful_fields() -> None:
    match = make_match("Paris FC", "Strasbourg", {"home": 2, "away": 1}, "FINISHED")
    match["score"]["halfTime"] = {"home": 1, "away": 0}

    assert summarize_match(match) == {
        "date": "2026-09-19",
        "competition": "FL1",
        "matchday": 5,
        "home": "Paris FC",
        "away": "Strasbourg",
        "score": "2-1",
        "half_time": "1-0",
        "duration": "REGULAR",
        "status": "FINISHED",
    }


def test_summarize_match_of_unplayed_match_has_no_score() -> None:
    match = make_match("PSG", "Monaco", {"home": None, "away": None}, "TIMED")

    summary = summarize_match(match)

    assert summary["score"] is None
    assert summary["half_time"] is None
    assert summary["status"] == "TIMED"


def test_summarize_matches_summarizes_every_match() -> None:
    raw = {
        "matches": [
            make_match("Paris FC", "Strasbourg", {"home": 2, "away": 1}, "FINISHED"),
            make_match("PSG", "Monaco", {"home": None, "away": None}, "TIMED"),
        ]
    }

    summaries = summarize_matches(raw)

    assert [(s["home"], s["away"]) for s in summaries] == [
        ("Paris FC", "Strasbourg"),
        ("PSG", "Monaco"),
    ]


def test_normalize_name_ignores_case_accents_and_surrounding_spaces() -> None:
    assert normalize_name("  Saint-Étienne ") == "saint-etienne"


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("PSG", 524),
        ("paris saint-germain fc", 524),
        ("Paris FC", 1045),
        ("saint-etienne", 527),
        ("Lyon", 523),
        ("Milan", 98),
    ],
)
def test_find_team_returns_the_only_matching_team(query: str, expected_id: int) -> None:
    assert find_team(TEAMS, query)["id"] == expected_id


def test_find_team_rejects_ambiguous_name_and_lists_candidates() -> None:
    with pytest.raises(ValueError, match="several teams: PSG, Paris FC"):
        find_team(TEAMS, "Paris")


def test_find_team_rejects_unknown_name_and_lists_available_teams() -> None:
    with pytest.raises(ValueError, match="No team matches 'OM'. Available teams: PSG, Paris FC"):
        find_team(TEAMS, "OM")


@pytest.mark.parametrize(
    ("winner", "team_id", "expected"),
    [
        ("HOME_TEAM", 1045, "W"),
        ("HOME_TEAM", 576, "L"),
        ("AWAY_TEAM", 576, "W"),
        ("DRAW", 1045, "D"),
    ],
)
def test_match_result_is_seen_from_the_given_team(winner: str, team_id: int, expected: str) -> None:
    match = make_match("Paris FC", "Strasbourg", {"home": 0, "away": 0}, "FINISHED")
    match["score"]["winner"] = winner

    assert match_result(match, team_id) == expected


def test_summarize_team_form_builds_form_in_match_order() -> None:
    win = make_match("Paris FC", "Strasbourg", {"home": 2, "away": 1}, "FINISHED")
    win["score"]["winner"] = "HOME_TEAM"
    loss = make_match("Paris FC", "Monaco", {"home": 0, "away": 3}, "FINISHED")
    loss["score"]["winner"] = "AWAY_TEAM"
    paris_fc = {"id": 1045, "name": "Paris FC", "shortName": "Paris FC"}

    form = summarize_team_form({"matches": [win, loss]}, paris_fc)

    assert form["team"] == "Paris FC"
    assert form["form"] == "WL"
    assert [match["result"] for match in form["matches"]] == ["W", "L"]
    assert form["matches"][0]["score"] == "2-1"
