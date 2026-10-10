from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from football_agent.agent_eval.scoring import Fact
from football_agent.predictor.data import LEAGUE_FILES
from football_agent.tools import DEFAULT_FORM_MATCHES, ToolContext, call_tool

COMPETITION_NAMES = {
    "PL": "Premier League",
    "FL1": "Ligue 1",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "PD": "La Liga",
    "CL": "Champions League",
}
# Teams are picked by their place in the table, a different top and bottom pair for each kind
# of question: many teams are covered and the same snapshot always gives the same questions.
TABLE_PLACES = {
    "position": (1, -2),
    "form_points": (0, -1),
    "last_result": (2, -3),
    "next_match": (3, -4),
}
QUESTIONS = {
    "leader": "Who is top of the {name}?",
    "position": "What position are {team} in the {name} table, and how many points do they have?",
    "form_points": "How many points have {team} taken from their last {last_n} {name} matches?",
    "last_result": "What was the score of {team}'s last {name} match?",
    "next_match": "Who do {team} play next in the {name}?",
    "day_results": "What were the {name} results on {day}?",
    "prediction": "Who is more likely to win, {home} or {away}, in their {name} match on {day}?",
}
# The first tool of each kind is the reference one: the evaluation runs it to find the answer.
ACCEPTED_TOOLS = {
    "leader": ("get_standings",),
    "position": ("get_standings",),
    "form_points": ("get_team_form", "get_matches"),
    "last_result": ("get_team_form", "get_matches"),
    "next_match": ("get_matches",),
    "day_results": ("get_matches",),
    "prediction": ("predict_match",),
}
UPCOMING = ("SCHEDULED", "TIMED")
NEXT_MATCH_WINDOW_DAYS = 30
PREDICTIONS_PER_LEAGUE = 2
ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth",
]  # fmt: skip


@dataclass(frozen=True)
class EvalCase:
    """A question and the arguments of the reference tool call whose result holds its answer."""

    kind: str
    question: str
    competition: str
    reference_arguments: dict[str, Any]
    team: str | None = None


def make_case(
    kind: str,
    competition: str,
    arguments: dict[str, Any],
    team: str | None = None,
    **details: Any,
) -> EvalCase:
    name = COMPETITION_NAMES[competition]
    question = QUESTIONS[kind].format(name=name, team=team, **details)
    return EvalCase(kind, question, competition, arguments, team)


def readable_date(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B} {day.year}"


def last_match_day(matches: list[dict[str, Any]], today: date) -> date:
    days = [date.fromisoformat(m["utcDate"][:10]) for m in matches if m["status"] == "FINISHED"]
    return max(day for day in days if day < today)


def next_fixtures(matches: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    upcoming = [
        match
        for match in matches
        if match["status"] in UPCOMING and date.fromisoformat(match["utcDate"][:10]) > today
    ]
    return sorted(upcoming, key=lambda match: match["utcDate"])[:PREDICTIONS_PER_LEAGUE]


def competition_cases(competition: str, recorded: dict[str, Any], today: date) -> list[EvalCase]:
    table = recorded["standings"]["standings"][0]["table"]
    matches = recorded["matches"]["matches"]
    teams = {
        kind: [table[place]["team"]["shortName"] for place in places]
        for kind, places in TABLE_PLACES.items()
    }
    standings = {"competition": competition}
    cases = [make_case("leader", competition, standings)]
    for team in teams["position"]:
        cases.append(make_case("position", competition, standings, team))
    for team in teams["last_result"]:
        form = {"team_name": team, "competition": competition, "last_n": 1}
        cases.append(make_case("last_result", competition, form, team))
    next_window = {
        "competition": competition,
        "date_from": today.isoformat(),
        "date_to": (today + timedelta(days=NEXT_MATCH_WINDOW_DAYS)).isoformat(),
    }
    for team in teams["next_match"]:
        cases.append(make_case("next_match", competition, next_window, team))
    day = last_match_day(matches, today)
    one_day = {"competition": competition, "date_from": day.isoformat(), "date_to": day.isoformat()}
    cases.append(make_case("day_results", competition, one_day, day=readable_date(day)))
    # The predictor only knows the leagues, and a Champions League team has too few matches in
    # the competition for its last five to make sense early in the season.
    if competition not in LEAGUE_FILES:
        return cases
    for team in teams["form_points"]:
        form = {"team_name": team, "competition": competition, "last_n": DEFAULT_FORM_MATCHES}
        cases.append(make_case("form_points", competition, form, team, last_n=DEFAULT_FORM_MATCHES))
    for match in next_fixtures(matches, today):
        home, away = match["homeTeam"]["shortName"], match["awayTeam"]["shortName"]
        kickoff = date.fromisoformat(match["utcDate"][:10])
        prediction = {
            "home_team": home,
            "away_team": away,
            "competition": competition,
            "match_date": kickoff.isoformat(),
        }
        day_text = readable_date(kickoff)
        cases.append(
            make_case("prediction", competition, prediction, home=home, away=away, day=day_text)
        )
    return cases


def build_cases(snapshot: dict[str, Any]) -> list[EvalCase]:
    """Write the evaluation questions from templates, about the teams and matches of a snapshot,
    as asked on the day it was recorded."""
    today = date.fromisoformat(snapshot["date"])
    return [
        case
        for competition, recorded in snapshot["competitions"].items()
        for case in competition_cases(competition, recorded, today)
    ]


def ordinal(number: int) -> str:
    """'1st', '2nd', '3rd', '4th', but '11th', '12th', '13th', and again '21st'."""
    if number % 100 in (11, 12, 13):
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


# A bare small number such as "3" appears in most answers, so positions and points are only
# accepted with the words around them.
def position_fact(position: int) -> Fact:
    spellings = [ordinal(position), f"position {position}", f"#{position}"]
    if position <= len(ORDINAL_WORDS):
        spellings.append(ORDINAL_WORDS[position - 1])
    return Fact(f"position {position}", tuple(spellings))


def points_fact(points: int) -> Fact:
    return Fact(
        f"{points} points",
        (f"{points} points", f"{points} point", f"{points} pts", f"| {points} |"),
    )


def team_fact(team: str, full_names: dict[str, str]) -> Fact:
    return Fact(team, (team, full_names[team]))


def score_fact(score: str, description: str) -> Fact:
    # "2-1" for the home team is "1-2" when the answer starts with the away team.
    home, away = score.split("-")
    return Fact(description, (score, f"{away}-{home}"))


def probability_fact(description: str, probability: float) -> Fact:
    percent = probability * 100
    return Fact(
        description,
        (
            f"{percent:.1f}%",
            f"{percent:.0f}%",
            f"{percent:.1f} percent",
            f"{percent:.0f} percent",
            f"{probability:.3f}",
            f"{probability:.2f}",
        ),
    )


def opponent(match: dict[str, Any], team: str) -> str:
    return match["away"] if match["home"] == team else match["home"]


def leader_facts(standings: dict[str, Any], full_names: dict[str, str]) -> list[Fact]:
    return [team_fact(standings["table"][0]["team"], full_names)]


def position_facts(standings: dict[str, Any], team: str) -> list[Fact]:
    row = next(row for row in standings["table"] if row["team"] == team)
    return [position_fact(row["position"]), points_fact(row["points"])]


def form_points_facts(form: dict[str, Any]) -> list[Fact]:
    return [points_fact(form["points"])]


def last_result_facts(form: dict[str, Any], full_names: dict[str, str]) -> list[Fact]:
    match = form["matches"][-1]
    return [
        score_fact(match["score"], "score of the last match"),
        team_fact(opponent(match, form["team"]), full_names),
    ]


def next_match_facts(
    matches: list[dict[str, Any]], team: str, full_names: dict[str, str]
) -> list[Fact]:
    upcoming = [m for m in matches if m["status"] in UPCOMING and team in (m["home"], m["away"])]
    if not upcoming:
        raise ValueError(f"{team} have no match in the reference window.")
    return [team_fact(opponent(min(upcoming, key=lambda m: m["date"]), team), full_names)]


def day_results_facts(matches: list[dict[str, Any]]) -> list[Fact]:
    return [
        score_fact(match["score"], f"score of {match['home']} vs {match['away']}")
        for match in matches
        if match["status"] == "FINISHED"
    ]


def prediction_facts(prediction: dict[str, Any]) -> list[Fact]:
    return [
        probability_fact(f"{prediction['home_team']} win probability", prediction["home_win"]),
        probability_fact(f"{prediction['away_team']} win probability", prediction["away_win"]),
    ]


def expected_facts(case: EvalCase, context: ToolContext) -> list[Fact]:
    """Run the reference tool call on the evaluation data and read in its result the facts that a
    correct answer must state: no expected answer is written by hand, so a new snapshot or a new
    champion model needs no rewrite."""
    result = call_tool(context, ACCEPTED_TOOLS[case.kind][0], case.reference_arguments)
    teams = context.football.get_teams(case.competition)["teams"]
    full_names = {team["shortName"]: team["name"] for team in teams}
    match case.kind:
        case "leader":
            return leader_facts(result, full_names)
        case "position":
            return position_facts(result, case.team)
        case "form_points":
            return form_points_facts(result)
        case "last_result":
            return last_result_facts(result, full_names)
        case "next_match":
            return next_match_facts(result, case.team, full_names)
        case "day_results":
            return day_results_facts(result)
        case "prediction":
            return prediction_facts(result)
    raise ValueError(f"Unknown kind of case '{case.kind}'.")
