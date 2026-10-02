from typing import Any

from football_agent.tools import (
    format_score,
    summarize_match,
    summarize_matches,
    summarize_standings,
)


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
