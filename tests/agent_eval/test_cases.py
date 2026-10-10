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

PSG, LYON, NICE, LENS = 524, 523, 522, 546
TEAMS = {
    PSG: ("Paris Saint-Germain FC", "PSG"),
    LYON: ("Olympique Lyonnais", "Lyon"),
    NICE: ("OGC Nice", "Nice"),
    LENS: ("Racing Club de Lens", "Lens"),
}


def team(team_id: int) -> dict[str, Any]:
    name, short_name = TEAMS[team_id]
    return {"id": team_id, "name": name, "shortName": short_name}


def row(position: int, team_id: int, points: int) -> dict[str, Any]:
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
    competition: str, day: str, home: int, away: int, score: tuple[int, int] | None = None
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
        "matchday": 1,
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


def recorded(competition: str, name: str) -> dict[str, Any]:
    matches = [
        match(competition, "2026-09-27", PSG, LYON, (2, 1)),
        match(competition, "2026-09-27", NICE, LENS, (0, 0)),
        match(competition, "2026-10-04", LYON, NICE, (1, 3)),
        match(competition, "2026-10-04", LENS, PSG, (2, 0)),
        match(competition, "2026-10-18", PSG, NICE),
        match(competition, "2026-10-18", LYON, LENS),
        match(competition, "2026-10-25", NICE, LYON),
    ]
    table = [row(1, NICE, 4), row(2, LENS, 4), row(3, PSG, 3), row(4, LYON, 0)]
    return {
        "standings": {
            "competition": {"name": name},
            "season": {"startDate": "2026-08-14"},
            "standings": [{"type": "TOTAL", "table": table}],
        },
        "teams": {"teams": [team(team_id) for team_id in TEAMS]},
        # Recorded in an order the cases must not rely on.
        "matches": {"matches": list(reversed(matches))},
    }


SNAPSHOT = {
    "date": "2026-10-09",
    "competitions": {
        "FL1": recorded("FL1", "Ligue 1"),
        "CL": recorded("CL", "UEFA Champions League"),
    },
}

# Every Ligue 1 question of the snapshot, with a correct answer.
ANSWERS = {
    "Who is top of the Ligue 1?": "**Nice** are top of Ligue 1.",
    "What position are Lens in the Ligue 1 table, and how many points do they have?": (
        "Lens are 2nd with 4 points."
    ),
    "What position are PSG in the Ligue 1 table, and how many points do they have?": (
        "PSG are third, on 3 pts."
    ),
    "What was the score of PSG's last Ligue 1 match?": "PSG lost 0-2 at Lens.",
    "What was the score of Lens's last Ligue 1 match?": "Lens beat Paris Saint-Germain FC 2 – 0.",
    "Who do Lyon play next in the Ligue 1?": "Lyon host Lens on Sunday 18 October.",
    "Who do Nice play next in the Ligue 1?": "Nice travel to PSG.",
    "What were the Ligue 1 results on Sunday 4 October 2026?": "Lyon 1-3 Nice\nLens 2-0 PSG",
    "How many points have Nice taken from their last 5 Ligue 1 matches?": "Nice took 4 points.",
    "How many points have Lyon taken from their last 5 Ligue 1 matches?": "Lyon took 0 points.",
    "Who is more likely to win, PSG or Nice, in their Ligue 1 match on Sunday 18 October 2026?": (
        "PSG: 50%, draw: 30%, Nice: 20%."
    ),
    "Who is more likely to win, Lyon or Lens, in their Ligue 1 match on Sunday 18 October 2026?": (
        "Lyon 50.0 percent, Lens 20.0 percent."
    ),
}


@pytest.fixture
def context(fake_predictor: Any) -> ToolContext:
    return ToolContext(SnapshotClient(SNAPSHOT), fake_predictor, date(2026, 10, 9))


def cases_by_question() -> dict[str, EvalCase]:
    return {case.question: case for case in build_cases(SNAPSHOT)}


def test_build_cases_asks_every_kind_of_question_on_a_league() -> None:
    questions = {case.question for case in build_cases(SNAPSHOT) if case.competition == "FL1"}

    assert questions == set(ANSWERS)


def test_build_cases_asks_no_form_or_prediction_question_on_the_champions_league() -> None:
    kinds = {case.kind for case in build_cases(SNAPSHOT) if case.competition == "CL"}

    assert kinds == {"leader", "position", "last_result", "next_match", "day_results"}


def test_build_cases_always_gives_the_same_questions() -> None:
    assert build_cases(SNAPSHOT) == build_cases(SNAPSHOT)


def test_build_cases_asks_about_the_last_day_with_results_before_the_snapshot() -> None:
    case = cases_by_question()["What were the Ligue 1 results on Sunday 4 October 2026?"]

    assert case.reference_arguments == {
        "competition": "FL1",
        "date_from": "2026-10-04",
        "date_to": "2026-10-04",
    }


def test_build_cases_predicts_the_next_fixture_on_its_date() -> None:
    question = (
        "Who is more likely to win, PSG or Nice, in their Ligue 1 match on Sunday 18 October 2026?"
    )

    assert cases_by_question()[question].reference_arguments["match_date"] == "2026-10-18"


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
