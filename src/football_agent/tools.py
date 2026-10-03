import unicodedata
from datetime import date
from typing import Any

from football_agent.data_api import FootballDataClient


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


def get_standings(
    client: FootballDataClient, competition: str, season: int | None = None
) -> list[dict[str, Any]]:
    return summarize_standings(client.get_standings(competition, season))


def get_matches(
    client: FootballDataClient, competition: str, date_from: str, date_to: str
) -> list[dict[str, Any]]:
    raw = client.get_matches(
        competition, date.fromisoformat(date_from), date.fromisoformat(date_to)
    )
    return summarize_matches(raw)


def get_team_form(
    client: FootballDataClient, team_name: str, competition: str, last_n: int = 5
) -> dict[str, Any]:
    team = find_team(client.get_teams(competition)["teams"], team_name)
    raw = client.get_team_matches(team["id"], competition, limit=last_n)
    return summarize_team_form(raw, team)


TOOL_FUNCTIONS = {
    "get_standings": get_standings,
    "get_matches": get_matches,
    "get_team_form": get_team_form,
}

COMPETITION_PARAMETER = {
    "type": "string",
    "enum": ["PL", "FL1", "BL1", "SA", "PD", "CL"],
    "description": (
        "Competition code: PL = Premier League (England), FL1 = Ligue 1 (France), "
        "BL1 = Bundesliga (Germany), SA = Serie A (Italy), PD = La Liga (Spain), "
        "CL = UEFA Champions League."
    ),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_standings",
            "description": (
                "Get the league table of a competition: position, points, wins, draws, losses "
                "and goals of every team. For the Champions League, this is the league phase table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "competition": COMPETITION_PARAMETER,
                    "season": {
                        "type": "integer",
                        "description": (
                            "Starting year of the season, e.g. 2025 for the 2025-26 season. "
                            "Omit it for the current season."
                        ),
                    },
                },
                "required": ["competition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_matches",
            "description": (
                "Get the matches of a competition between two dates, both included: "
                "final and half-time scores of finished matches, status of upcoming ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "competition": COMPETITION_PARAMETER,
                    "date_from": {"type": "string", "description": "First day, YYYY-MM-DD."},
                    "date_to": {"type": "string", "description": "Last day, YYYY-MM-DD."},
                },
                "required": ["competition", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_form",
            "description": (
                "Get the last results of a team in a competition: a form string such as 'WWDLW' "
                "(W = win, D = draw, L = loss, oldest match first) and the detail of each match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": (
                            "Team name, e.g. 'PSG', 'Liverpool', 'Bayern'. If it is ambiguous or "
                            "unknown, the error message lists the valid team names."
                        ),
                    },
                    "competition": COMPETITION_PARAMETER,
                    "last_n": {
                        "type": "integer",
                        "description": "Number of last finished matches to return. Defaults to 5.",
                    },
                },
                "required": ["team_name", "competition"],
            },
        },
    },
]


def call_tool(
    client: FootballDataClient, name: str, arguments: dict[str, Any]
) -> list[dict[str, Any]] | dict[str, Any]:
    """Run the tool called `name` with the arguments chosen by the LLM."""
    if name not in TOOL_FUNCTIONS:
        available = ", ".join(TOOL_FUNCTIONS)
        raise ValueError(f"Unknown tool '{name}'. Available tools: {available}.")
    return TOOL_FUNCTIONS[name](client, **arguments)
