from typing import Any


def format_score(score: dict[str, int | None]) -> str | None:
    if score["home"] is None:
        return None
    return f"{score['home']}-{score['away']}"


def summarize_standings(raw: dict[str, Any]) -> list[dict[str, Any]]:
    # Leagues and the Champions League league phase both come back as a single total table.
    table = raw["standings"][0]["table"]
    return [
        {
            "position": row["position"],
            "team": row["team"]["shortName"],
            "played": row["playedGames"],
            "won": row["won"],
            "draw": row["draw"],
            "lost": row["lost"],
            "goals_for": row["goalsFor"],
            "goals_against": row["goalsAgainst"],
            "goal_difference": row["goalDifference"],
            "points": row["points"],
        }
        for row in table
    ]


def summarize_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": match["utcDate"][:10],
        "competition": match["competition"]["code"],
        "matchday": match["matchday"],
        "home": match["homeTeam"]["shortName"],
        "away": match["awayTeam"]["shortName"],
        "score": format_score(match["score"]["fullTime"]),
        "half_time": format_score(match["score"]["halfTime"]),
        "duration": match["score"]["duration"],
        "status": match["status"],
    }


def summarize_matches(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [summarize_match(match) for match in raw["matches"]]
