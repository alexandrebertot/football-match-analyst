import unicodedata
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


def normalize_name(name: str) -> str:
    """Lowercase and strip accents and surrounding spaces, so that ' Étienne' equals 'etienne'."""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.casefold().strip()


def find_team(teams: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Return the only team matching `name`, exactly first, then as a substring.

    Raise ValueError listing the candidates when no team or several teams match,
    so that the LLM can retry with a valid name.
    """
    wanted = normalize_name(name)
    team_names = [
        (team, normalize_name(team["name"]), normalize_name(team["shortName"])) for team in teams
    ]

    exact = [team for team, full, short in team_names if wanted in (full, short)]
    if len(exact) == 1:
        return exact[0]

    partial = [team for team, full, short in team_names if wanted in full or wanted in short]
    if len(partial) == 1:
        return partial[0]
    if partial:
        candidates = ", ".join(team["shortName"] for team in partial)
        raise ValueError(f"'{name}' matches several teams: {candidates}. Use one of these names.")
    available = ", ".join(team["shortName"] for team in teams)
    raise ValueError(f"No team matches '{name}'. Available teams: {available}.")


def match_result(match: dict[str, Any], team_id: int) -> str:
    """Return 'W', 'D' or 'L' for the team `team_id` in a finished match."""
    winner = match["score"]["winner"]
    if winner == "DRAW":
        return "D"
    team_side = "HOME_TEAM" if match["homeTeam"]["id"] == team_id else "AWAY_TEAM"
    return "W" if winner == team_side else "L"


def summarize_team_form(raw: dict[str, Any], team: dict[str, Any]) -> dict[str, Any]:
    matches = [
        {**summarize_match(match), "result": match_result(match, team["id"])}
        for match in raw["matches"]
    ]
    return {
        "team": team["shortName"],
        "form": "".join(match["result"] for match in matches),
        "matches": matches,
    }
