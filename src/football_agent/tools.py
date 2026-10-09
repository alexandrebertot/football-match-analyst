import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from football_agent.predictor.data import LEAGUE_FILES
from football_agent.predictor.predict import MatchPredictor


def format_score(score: dict[str, int | None]) -> str | None:
    if score["home"] is None:
        return None
    return f"{score['home']}-{score['away']}"


def season_label(start_date: str) -> str:
    """Return the season label, e.g. '2026-27' for a season starting on '2026-08-22'."""
    start_year = date.fromisoformat(start_date).year
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def summarize_standings(raw: dict[str, Any]) -> dict[str, Any]:
    # Leagues and the Champions League league phase both come back as a single total table.
    table = raw["standings"][0]["table"]
    return {
        "competition": raw["competition"]["name"],
        "season": season_label(raw["season"]["startDate"]),
        "table": [
            {
                "position": row["position"],
                "team": row["team"]["shortName"],
                "played": row["playedGames"],
                "won": row["won"],
                "drawn": row["draw"],
                "lost": row["lost"],
                "goals_for": row["goalsFor"],
                "goals_against": row["goalsAgainst"],
                "goal_difference": row["goalDifference"],
                "points": row["points"],
            }
            for row in table
        ],
    }


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def summarize_match(match: dict[str, Any]) -> dict[str, Any]:
    kickoff_date = date.fromisoformat(match["utcDate"][:10])
    return {
        "date": kickoff_date.isoformat(),
        "weekday": WEEKDAYS[kickoff_date.weekday()],
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


DEFAULT_FORM_MATCHES = 5
RESULT_POINTS = {"W": 3, "D": 1, "L": 0}


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
    results = [match["result"] for match in matches]
    return {
        "team": team["shortName"],
        "played": len(results),
        "won": results.count("W"),
        "drawn": results.count("D"),
        "lost": results.count("L"),
        "points": sum(RESULT_POINTS[result] for result in results),
        "form": "".join(results),
        "matches": matches,
    }


class FootballData(Protocol):
    """The football data the tools read: the live API client, or a recorded snapshot."""

    def get_standings(self, competition: str, season: int | None = None) -> dict[str, Any]: ...

    def get_teams(self, competition: str) -> dict[str, Any]: ...

    def get_matches(self, competition: str, date_from: date, date_to: date) -> dict[str, Any]: ...

    def get_team_matches(self, team_id: int, competition: str, limit: int) -> dict[str, Any]: ...


@dataclass
class ToolContext:
    """What the tools need besides the arguments chosen by the LLM."""

    football: FootballData
    predictor: MatchPredictor
    today: date


def get_standings(
    context: ToolContext, competition: str, season: int | None = None
) -> dict[str, Any]:
    return summarize_standings(context.football.get_standings(competition, season))


def get_matches(
    context: ToolContext, competition: str, date_from: str, date_to: str
) -> list[dict[str, Any]]:
    raw = context.football.get_matches(
        competition, date.fromisoformat(date_from), date.fromisoformat(date_to)
    )
    return summarize_matches(raw)


def get_team_form(
    context: ToolContext, team_name: str, competition: str, last_n: int = DEFAULT_FORM_MATCHES
) -> dict[str, Any]:
    team = find_team(context.football.get_teams(competition)["teams"], team_name)
    raw = context.football.get_team_matches(team["id"], competition, limit=last_n)
    return summarize_team_form(raw, team)


def predict_match(
    context: ToolContext,
    home_team: str,
    away_team: str,
    competition: str,
    match_date: str | None = None,
) -> dict[str, Any]:
    teams = context.football.get_teams(competition)["teams"]
    home, away = find_team(teams, home_team), find_team(teams, away_team)
    if home["id"] == away["id"]:
        raise ValueError(f"'{home_team}' and '{away_team}' are the same team.")
    kickoff = date.fromisoformat(match_date) if match_date else context.today
    prediction = context.predictor.predict(home, away, kickoff)
    return {"home_team": home["shortName"], "away_team": away["shortName"], **prediction}


TOOL_FUNCTIONS = {
    "get_standings": get_standings,
    "get_matches": get_matches,
    "get_team_form": get_team_form,
    "predict_match": predict_match,
}

COMPETITIONS = [*LEAGUE_FILES, "CL"]
LEAGUES_DESCRIPTION = (
    "PL = Premier League (England), FL1 = Ligue 1 (France), BL1 = Bundesliga (Germany), "
    "SA = Serie A (Italy), PD = La Liga (Spain)"
)
COMPETITION_PARAMETER = {
    "type": "string",
    "enum": COMPETITIONS,
    "description": f"Competition code: {LEAGUES_DESCRIPTION}, CL = UEFA Champions League.",
}
LEAGUE_PARAMETER = {
    "type": "string",
    "enum": list(LEAGUE_FILES),
    "description": f"League code: {LEAGUES_DESCRIPTION}. The Champions League is not supported.",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_standings",
            "description": (
                "Get the league table of a competition and the season it belongs to (e.g. "
                "'2026-27'): position, points, wins, draws, losses and goals of every team. "
                "For the Champions League, this is the league phase table."
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
                "Get the matches of a competition between two dates, both included: date and "
                "weekday, final and half-time scores of finished matches, status of upcoming ones."
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
                "Get the last results of a team in a competition: matches played, won, drawn and "
                "lost, points earned, a form string such as 'WWDLW' (W = win, D = draw, L = loss, "
                "oldest match first) and the detail of each match."
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
                        "description": (
                            "Number of last finished matches to return. "
                            f"Defaults to {DEFAULT_FORM_MATCHES}."
                        ),
                    },
                },
                "required": ["team_name", "competition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_match",
            "description": (
                "Predict the outcome of a league match with a machine learning model: "
                "probabilities of a home win, a draw and an away win, plus, for each team, the "
                "recent form the model used: averages over its last `recent_form_matches` "
                "matches, home and away games combined. These are estimates, not certainties: "
                "present them as such."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string", "description": "Home team name, e.g. 'PSG'."},
                    "away_team": {
                        "type": "string",
                        "description": "Away team name, e.g. 'Marseille'.",
                    },
                    "competition": LEAGUE_PARAMETER,
                    "match_date": {
                        "type": "string",
                        "description": "Match date, YYYY-MM-DD. Omit it for today.",
                    },
                },
                "required": ["home_team", "away_team", "competition"],
            },
        },
    },
]


def call_tool(
    context: ToolContext, name: str, arguments: dict[str, Any]
) -> list[dict[str, Any]] | dict[str, Any]:
    """Run the tool called `name` with the arguments chosen by the LLM."""
    if name not in TOOL_FUNCTIONS:
        available = ", ".join(TOOL_FUNCTIONS)
        raise ValueError(f"Unknown tool '{name}'. Available tools: {available}.")
    return TOOL_FUNCTIONS[name](context, **arguments)
