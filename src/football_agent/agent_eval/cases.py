from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from football_agent.agent_eval.scoring import Fact
from football_agent.predictor.data import LEAGUE_FILES
from football_agent.tools import DEFAULT_FORM_MATCHES, ToolContext, call_tool

COMPETITION_NAMES = {
    "en": {
        "PL": "Premier League",
        "FL1": "Ligue 1",
        "BL1": "Bundesliga",
        "SA": "Serie A",
        "PD": "La Liga",
        "CL": "Champions League",
    },
    "fr": {
        "PL": "Premier League",
        "FL1": "Ligue 1",
        "BL1": "Bundesliga",
        "SA": "Serie A",
        "PD": "Liga",
        "CL": "Ligue des champions",
    },
}
# Teams are picked by their place in the table, a different top and bottom pair for each kind
# of question: many teams are covered and the same snapshot always gives the same questions.
TABLE_PLACES = {
    "position": (1, -2),
    "form_points": (0, -1),
    "last_result": (2, -3),
    "next_match": (3, -4),
}
COMPARISON_PLACES = (4, -5)
# Places among the teams of a league that do not play the Champions League, so that a question
# without its competition has a single right answer.
LEAGUE_ONLY_PLACES = (1, -2)
QUESTIONS = {
    "en": {
        "leader": "Who is top of the {name}?",
        "position": (
            "What position are {team} in the {name} table, and how many points do they have?"
        ),
        "position_no_competition": (
            "Where are {team} in the table, and how many points do they have?"
        ),
        "form_points": (
            "How many points have {team} taken from their last {last_n} {name} matches?"
        ),
        "form_comparison": (
            "How many points have {team} and {other} each taken from their last {last_n} {name} "
            "matches?"
        ),
        "last_result": "What was the score of {team}'s last {name} match, and who did they play?",
        "last_result_no_competition": (
            "What was the score of {team}'s last match, and who did they play?"
        ),
        "next_match": "Who do {team} play next in the {name}?",
        "day_results": "What were the {name} results on {day}?",
        "matchday_results": "What were the {name} results of matchday {matchday}?",
        "prediction": (
            "Who is more likely to win, {home} or {away}, in their {name} match on {day}?"
        ),
    },
    "fr": {
        "leader": "Qui est en tête de la {name} ?",
        "form_points": (
            "Combien de points a pris {team} lors de ses {last_n} derniers matchs de {name} ?"
        ),
        "last_result": "Quel a été le score du dernier match de {team} en {name}, et contre qui ?",
        "next_match": "Contre qui joue {team} lors de son prochain match de {name} ?",
        "prediction": (
            "Qui a le plus de chances de gagner entre {home} et {away}, lors de leur match de "
            "{name} du {day} ?"
        ),
    },
}
# The first tool of each kind is the reference one: the evaluation runs it to find the answer.
ACCEPTED_TOOLS = {
    "leader": ("get_standings",),
    "position": ("get_standings",),
    "position_no_competition": ("get_standings",),
    "form_points": ("get_team_form", "get_matches"),
    "form_comparison": ("get_team_form", "get_matches"),
    "last_result": ("get_team_form", "get_matches"),
    "last_result_no_competition": ("get_team_form", "get_matches"),
    "next_match": ("get_matches",),
    "day_results": ("get_matches",),
    "matchday_results": ("get_matches",),
    "prediction": ("predict_match",),
}
FRENCH_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
    "octobre", "novembre", "décembre",
]  # fmt: skip
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
    other_team: str | None = None
    language: str = "en"


def make_case(
    kind: str,
    competition: str,
    arguments: dict[str, Any],
    team: str | None = None,
    *,
    other_team: str | None = None,
    language: str = "en",
    **details: Any,
) -> EvalCase:
    name = COMPETITION_NAMES[language][competition]
    question = QUESTIONS[language][kind].format(name=name, team=team, other=other_team, **details)
    return EvalCase(kind, question, competition, arguments, team, other_team, language)


def readable_date(day: date, language: str = "en") -> str:
    if language == "fr":
        weekday, month = FRENCH_WEEKDAYS[day.weekday()], FRENCH_MONTHS[day.month - 1]
        return f"{weekday} {day.day} {month} {day.year}"
    return f"{day:%A} {day.day} {day:%B} {day.year}"


def last_match_day(matches: list[dict[str, Any]], today: date) -> date:
    days = [date.fromisoformat(m["utcDate"][:10]) for m in matches if m["status"] == "FINISHED"]
    return max(day for day in days if day < today)


def last_finished_matchday(matches: list[dict[str, Any]], today: date) -> int:
    """The last matchday whose matches were all played before `today`."""
    matchdays = {match["matchday"] for match in matches if match["matchday"] is not None}
    return max(
        matchday
        for matchday in matchdays
        if all(
            match["status"] == "FINISHED" and date.fromisoformat(match["utcDate"][:10]) < today
            for match in matches
            if match["matchday"] == matchday
        )
    )


def next_fixtures(matches: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    upcoming = [
        match
        for match in matches
        if match["status"] in UPCOMING and date.fromisoformat(match["utcDate"][:10]) > today
    ]
    return sorted(upcoming, key=lambda match: match["utcDate"])[:PREDICTIONS_PER_LEAGUE]


def picked_teams(table: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        kind: [table[place]["team"]["shortName"] for place in places]
        for kind, places in TABLE_PLACES.items()
    }


def form_arguments(team: str, competition: str, last_n: int) -> dict[str, Any]:
    return {"team_name": team, "competition": competition, "last_n": last_n}


def next_match_arguments(team: str, competition: str, today: date) -> dict[str, Any]:
    return {
        "competition": competition,
        "date_from": today.isoformat(),
        "date_to": (today + timedelta(days=NEXT_MATCH_WINDOW_DAYS)).isoformat(),
        "team_name": team,
    }


def prediction_case(match: dict[str, Any], competition: str, language: str = "en") -> EvalCase:
    home, away = match["homeTeam"]["shortName"], match["awayTeam"]["shortName"]
    kickoff = date.fromisoformat(match["utcDate"][:10])
    prediction = {
        "home_team": home,
        "away_team": away,
        "competition": competition,
        "match_date": kickoff.isoformat(),
    }
    day = readable_date(kickoff, language)
    return make_case(
        "prediction", competition, prediction, language=language, home=home, away=away, day=day
    )


def competition_cases(competition: str, recorded: dict[str, Any], today: date) -> list[EvalCase]:
    table = recorded["standings"]["standings"][0]["table"]
    matches = recorded["matches"]["matches"]
    teams = picked_teams(table)
    standings = {"competition": competition}
    cases = [make_case("leader", competition, standings)]
    for team in teams["position"]:
        cases.append(make_case("position", competition, standings, team))
    for team in teams["last_result"]:
        form = form_arguments(team, competition, 1)
        cases.append(make_case("last_result", competition, form, team))
    for team in teams["next_match"]:
        next_window = next_match_arguments(team, competition, today)
        cases.append(make_case("next_match", competition, next_window, team))
    day = last_match_day(matches, today)
    one_day = {"competition": competition, "date_from": day.isoformat(), "date_to": day.isoformat()}
    cases.append(make_case("day_results", competition, one_day, day=readable_date(day)))
    # The predictor only knows the leagues, and a Champions League team has too few matches in
    # the competition for its last five to make sense early in the season.
    if competition not in LEAGUE_FILES:
        return cases
    for team in teams["form_points"]:
        form = form_arguments(team, competition, DEFAULT_FORM_MATCHES)
        cases.append(make_case("form_points", competition, form, team, last_n=DEFAULT_FORM_MATCHES))
    for match in next_fixtures(matches, today):
        cases.append(prediction_case(match, competition))
    return cases


def harder_cases(
    competition: str, recorded: dict[str, Any], today: date, champions_league_ids: set[int]
) -> list[EvalCase]:
    """Questions closer to what users ask: in French, without the competition, about two teams,
    or about a matchday."""
    table = recorded["standings"]["standings"][0]["table"]
    matches = recorded["matches"]["matches"]
    teams = picked_teams(table)
    matchday = last_finished_matchday(matches, today)
    cases = [
        make_case(
            "matchday_results",
            competition,
            {"competition": competition, "matchday": matchday},
            matchday=matchday,
        ),
        # The same questions as the first English ones, so that the two languages compare paired.
        make_case("leader", competition, {"competition": competition}, language="fr"),
        make_case(
            "last_result",
            competition,
            form_arguments(teams["last_result"][0], competition, 1),
            teams["last_result"][0],
            language="fr",
        ),
        make_case(
            "next_match",
            competition,
            next_match_arguments(teams["next_match"][0], competition, today),
            teams["next_match"][0],
            language="fr",
        ),
    ]
    if competition not in LEAGUE_FILES:
        return cases
    cases += [
        make_case(
            "form_points",
            competition,
            form_arguments(teams["form_points"][0], competition, DEFAULT_FORM_MATCHES),
            teams["form_points"][0],
            language="fr",
            last_n=DEFAULT_FORM_MATCHES,
        ),
        prediction_case(next_fixtures(matches, today)[0], competition, language="fr"),
    ]
    team, other = (table[place]["team"]["shortName"] for place in COMPARISON_PLACES)
    cases.append(
        make_case(
            "form_comparison",
            competition,
            form_arguments(team, competition, DEFAULT_FORM_MATCHES),
            team,
            other_team=other,
            last_n=DEFAULT_FORM_MATCHES,
        )
    )
    league_only = [
        row["team"]["shortName"] for row in table if row["team"]["id"] not in champions_league_ids
    ]
    for team in [league_only[place] for place in LEAGUE_ONLY_PLACES]:
        cases.append(
            make_case("position_no_competition", competition, {"competition": competition}, team)
        )
        cases.append(
            make_case(
                "last_result_no_competition",
                competition,
                form_arguments(team, competition, 1),
                team,
            )
        )
    return cases


def build_cases(snapshot: dict[str, Any]) -> list[EvalCase]:
    """Write the evaluation questions from templates, about the teams and matches of a snapshot,
    as asked on the day it was recorded."""
    today = date.fromisoformat(snapshot["date"])
    champions_league_ids = {team["id"] for team in snapshot["competitions"]["CL"]["teams"]["teams"]}
    return [
        case
        for competition, recorded in snapshot["competitions"].items()
        for case in competition_cases(competition, recorded, today)
        + harder_cases(competition, recorded, today, champions_league_ids)
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
        case "position" | "position_no_competition":
            return position_facts(result, case.team)
        case "form_points":
            return form_points_facts(result)
        case "form_comparison":
            other_arguments = {**case.reference_arguments, "team_name": case.other_team}
            other = call_tool(context, "get_team_form", other_arguments)
            return form_points_facts(result) + form_points_facts(other)
        case "last_result" | "last_result_no_competition":
            return last_result_facts(result, full_names)
        case "next_match":
            return next_match_facts(result, case.team, full_names)
        case "day_results" | "matchday_results":
            return day_results_facts(result)
        case "prediction":
            return prediction_facts(result)
    raise ValueError(f"Unknown kind of case '{case.kind}'.")
