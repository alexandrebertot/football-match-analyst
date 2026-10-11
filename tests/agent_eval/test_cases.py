from datetime import date
from typing import Any

import pytest

from football_agent.agent_eval.cases import (
    EvalCase,
    build_cases,
    expected_facts,
    next_match_facts,
    ordinal,
    points_fact,
    position_fact,
    probability_fact,
    score_fact,
    team_fact,
)
from football_agent.agent_eval.scoring import fact_found
from football_agent.agent_eval.snapshot import SnapshotClient
from football_agent.tools import ToolContext

PSG, LYON, NICE, LENS, MONACO, MARSEILLE = 524, 523, 522, 546, 548, 516
TEAMS = {
    PSG: ("Paris Saint-Germain FC", "PSG"),
    LYON: ("Olympique Lyonnais", "Lyon"),
    NICE: ("OGC Nice", "Nice"),
    LENS: ("Racing Club de Lens", "Lens"),
    MONACO: ("AS Monaco FC", "Monaco"),
    MARSEILLE: ("Olympique de Marseille", "Marseille"),
}
CHAMPIONS_LEAGUE_TEAMS = [PSG, LYON, NICE, LENS]
POINTS = {NICE: 4, LENS: 4, MONACO: 4, PSG: 3, MARSEILLE: 1, LYON: 0}
# (matchday, day, home, away, score), the score of a match still to play being None.
SEASON = [
    (1, "2026-09-26", MONACO, MARSEILLE, (1, 1)),
    (1, "2026-09-27", PSG, LYON, (2, 1)),
    (1, "2026-09-27", NICE, LENS, (0, 0)),
    (2, "2026-10-03", MARSEILLE, MONACO, (0, 2)),
    (2, "2026-10-04", LYON, NICE, (1, 3)),
    (2, "2026-10-04", LENS, PSG, (2, 0)),
    (3, "2026-10-16", MONACO, MARSEILLE, None),
    (3, "2026-10-17", PSG, NICE, None),
    (3, "2026-10-18", LYON, LENS, None),
    (4, "2026-10-25", NICE, LYON, None),
]


def team(team_id: int) -> dict[str, Any]:
    name, short_name = TEAMS[team_id]
    return {"id": team_id, "name": name, "shortName": short_name}


def row(position: int, team_id: int) -> dict[str, Any]:
    points = POINTS[team_id]
    return {
        "position": position,
        "team": team(team_id),
        "playedGames": 2,
        "won": points // 3,
        "draw": points % 3,
        "lost": 2 - points // 3 - points % 3,
        "goalsFor": 2,
        "goalsAgainst": 2,
        "goalDifference": 0,
        "points": points,
    }


def match(
    competition: str,
    matchday: int,
    day: str,
    home: int,
    away: int,
    score: tuple[int, int] | None,
) -> dict[str, Any]:
    home_goals, away_goals = score or (None, None)
    winner = None
    if score:
        winner = "DRAW" if home_goals == away_goals else "HOME_TEAM"
        if home_goals < away_goals:
            winner = "AWAY_TEAM"
    return {
        "utcDate": f"{day}T19:00:00Z",
        "status": "FINISHED" if score else "TIMED",
        "matchday": matchday,
        "competition": {"code": competition},
        "homeTeam": team(home),
        "awayTeam": team(away),
        "score": {
            "fullTime": {"home": home_goals, "away": away_goals},
            "halfTime": {"home": home_goals, "away": away_goals},
            "duration": "REGULAR",
            "winner": winner,
        },
    }


def recorded(competition: str, name: str, team_ids: list[int]) -> dict[str, Any]:
    ranked = sorted(team_ids, key=lambda team_id: -POINTS[team_id])
    matches = [
        match(competition, *played)
        for played in SEASON
        if played[2] in team_ids and played[3] in team_ids
    ]
    return {
        "standings": {
            "competition": {"name": name},
            "season": {"startDate": "2026-08-14"},
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [row(place, team_id) for place, team_id in enumerate(ranked, 1)],
                }
            ],
        },
        "teams": {"teams": [team(team_id) for team_id in team_ids]},
        # Recorded in an order the cases must not rely on.
        "matches": {"matches": list(reversed(matches))},
    }


SNAPSHOT = {
    "date": "2026-10-09",
    "competitions": {
        "FL1": recorded("FL1", "Ligue 1", list(TEAMS)),
        "CL": recorded("CL", "UEFA Champions League", CHAMPIONS_LEAGUE_TEAMS),
    },
}

# Every Ligue 1 question of the snapshot, with a correct answer. The Ligue 1 table is Nice,
# Lens, Monaco, PSG, Marseille, Lyon; Monaco and Marseille do not play the Champions League.
ANSWERS = {
    "Who is top of the Ligue 1?": "**Nice** are top of Ligue 1.",
    "What position are Lens in the Ligue 1 table, and how many points do they have?": (
        "Lens are 2nd with 4 points."
    ),
    "What position are Marseille in the Ligue 1 table, and how many points do they have?": (
        "Marseille are fifth, on 1 point."
    ),
    "What was the score of Monaco's last Ligue 1 match, and who did they play?": (
        "Monaco won 2-0 at Marseille."
    ),
    "What was the score of PSG's last Ligue 1 match, and who did they play?": (
        "PSG lost 0-2 at Lens."
    ),
    "Who do PSG play next in the Ligue 1?": "PSG host Nice on Saturday.",
    "Who do Monaco play next in the Ligue 1?": "Monaco host Olympique de Marseille.",
    "What were the Ligue 1 results on Sunday 4 October 2026?": "Lyon 1-3 Nice\nLens 2-0 PSG",
    "How many points have Nice taken from their last 5 Ligue 1 matches?": "Nice took 4 points.",
    "How many points have Lyon taken from their last 5 Ligue 1 matches?": "Lyon took 0 points.",
    "Who is more likely to win, Monaco or Marseille, in their Ligue 1 match on Friday 16 October "
    "2026?": "Monaco: 50%, draw: 30%, Marseille: 20%.",
    "Who is more likely to win, PSG or Nice, in their Ligue 1 match on Saturday 17 October 2026?": (
        "PSG 50.0 percent, Nice 20.0 percent."
    ),
    "What were the Ligue 1 results of matchday 2?": (
        "Marseille 0-2 Monaco, Lyon 1-3 Nice and Lens 2-0 PSG."
    ),
    "How many points have Marseille and Lens each taken from their last 5 Ligue 1 matches?": (
        "Marseille took 1 point, Lens 4 points."
    ),
    "Where are Marseille in the table, and how many points do they have?": (
        "Marseille are 5th with 1 point."
    ),
    "Where are Monaco in the table, and how many points do they have?": (
        "Monaco are third on 4 pts."
    ),
    "What was the score of Marseille's last match, and who did they play?": (
        "Marseille lost 0-2 at home to Monaco."
    ),
    "What was the score of Monaco's last match, and who did they play?": (
        "Monaco beat Marseille 2-0."
    ),
    "Qui est en tête de la Ligue 1 ?": "**Nice** est en tête de la Ligue 1.",
    "Quel a été le score du dernier match de Monaco en Ligue 1, et contre qui ?": (
        "Monaco a gagné 2-0 à Marseille."
    ),
    "Contre qui joue PSG lors de son prochain match de Ligue 1 ?": "Le PSG reçoit Nice samedi.",
    "Combien de points a pris Nice lors de ses 5 derniers matchs de Ligue 1 ?": (
        "Nice a pris 4 points."
    ),
    "Qui a le plus de chances de gagner entre Monaco et Marseille, lors de leur match de Ligue 1 "
    "du vendredi 16 octobre 2026 ?": "Monaco : 50 %, nul : 30 %, Marseille : 20 %.",
}


@pytest.fixture
def context(fake_predictor: Any) -> ToolContext:
    return ToolContext(SnapshotClient(SNAPSHOT), fake_predictor, date(2026, 10, 9))


def cases_by_question() -> dict[str, EvalCase]:
    return {case.question: case for case in build_cases(SNAPSHOT)}


def test_build_cases_asks_every_kind_of_question_on_a_league() -> None:
    questions = {case.question for case in build_cases(SNAPSHOT) if case.competition == "FL1"}

    assert questions == set(ANSWERS)


def test_build_cases_asks_only_some_kinds_of_question_on_the_champions_league() -> None:
    kinds = {
        (case.kind, case.language) for case in build_cases(SNAPSHOT) if case.competition == "CL"
    }

    assert kinds == {
        ("leader", "en"),
        ("position", "en"),
        ("last_result", "en"),
        ("next_match", "en"),
        ("day_results", "en"),
        ("matchday_results", "en"),
        ("leader", "fr"),
        ("last_result", "fr"),
        ("next_match", "fr"),
    }


def test_build_cases_asks_each_french_question_in_english_too() -> None:
    cases = build_cases(SNAPSHOT)
    english = [(case.kind, case.reference_arguments) for case in cases if case.language == "en"]

    for case in cases:
        if case.language == "fr":
            assert (case.kind, case.reference_arguments) in english


def test_build_cases_always_gives_the_same_questions() -> None:
    assert build_cases(SNAPSHOT) == build_cases(SNAPSHOT)


def test_build_cases_asks_about_the_last_day_with_results_before_the_snapshot() -> None:
    case = cases_by_question()["What were the Ligue 1 results on Sunday 4 October 2026?"]

    assert case.reference_arguments == {
        "competition": "FL1",
        "date_from": "2026-10-04",
        "date_to": "2026-10-04",
    }


def test_build_cases_asks_about_the_last_matchday_played_before_the_snapshot() -> None:
    case = cases_by_question()["What were the Ligue 1 results of matchday 2?"]

    assert case.reference_arguments == {"competition": "FL1", "matchday": 2}


def test_build_cases_predicts_the_next_fixture_on_its_date() -> None:
    question = (
        "Who is more likely to win, PSG or Nice, in their Ligue 1 match on Saturday 17 October "
        "2026?"
    )

    assert cases_by_question()[question].reference_arguments["match_date"] == "2026-10-17"


@pytest.mark.parametrize("question", ANSWERS)
def test_expected_facts_are_all_in_a_correct_answer(question: str, context: ToolContext) -> None:
    facts = expected_facts(cases_by_question()[question], context)

    assert facts
    assert all(fact_found(fact, ANSWERS[question]) for fact in facts)
    assert not any(fact_found(fact, "Sorry, I could not find this.") for fact in facts)


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"),
     (21, "21st"), (22, "22nd"), (35, "35th")],
)  # fmt: skip
def test_ordinal(number: int, expected: str) -> None:
    assert ordinal(number) == expected


@pytest.mark.parametrize("answer", ["They are third.", "3rd place", "at position 3", "#3 Lens"])
def test_position_fact_accepts_a_position_in_words_or_figures(answer: str) -> None:
    assert fact_found(position_fact(3), answer)


@pytest.mark.parametrize("answer", ["| Lens | 13 |", "13 points", "13 pts"])
def test_points_fact_accepts_points_in_a_sentence_or_a_table(answer: str) -> None:
    assert fact_found(points_fact(13), answer)


def test_small_numbers_need_their_context() -> None:
    answer = "Lens won 3 of their 13 matches."

    assert not fact_found(position_fact(3), answer)
    assert not fact_found(points_fact(13), answer)


def test_team_fact_accepts_the_short_and_the_full_name() -> None:
    fact = team_fact("PSG", {"PSG": "Paris Saint-Germain FC"})

    assert fact_found(fact, "PSG lead.")
    assert fact_found(fact, "Paris Saint-Germain FC lead.")


@pytest.mark.parametrize("answer", ["Lens won 2-1.", "Nice lost 1-2 at Lens."])
def test_score_fact_accepts_both_orientations(answer: str) -> None:
    assert fact_found(score_fact("2-1", "score"), answer)


@pytest.mark.parametrize("answer", ["57.9%", "58%", "57.9 percent", "0.579", "0.58"])
def test_probability_fact_accepts_percentages_and_fractions(answer: str) -> None:
    assert fact_found(probability_fact("home win", 0.579), f"Home win: {answer}.")


def test_next_match_facts_fails_loudly_without_an_upcoming_match() -> None:
    with pytest.raises(ValueError, match="PSG have no match"):
        next_match_facts([], "PSG", {})
