import pytest

from football_agent.agent_eval.scoring import (
    Fact,
    ToolCall,
    called_expected_tool,
    fact_found,
    normalize_text,
    score_case,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Saint-Étienne", "saint-etienne"),
        ("57,9 %", "57.9%"),
        ("57,9\u00a0%", "57.9%"),
        ("M’gladbach", "m'gladbach"),
        ("won 2 – 1", "won 2-1"),
        ("won 2:1", "won 2-1"),
        ("  Man   City ", "man city"),
    ],
)
def test_normalize_text_makes_spellings_comparable(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    "answer", ["They have a 57.9% chance.", "Home win: 57,9 %", "about 58% to win"]
)
def test_fact_found_accepts_any_listed_spelling(answer: str) -> None:
    home_win = Fact("home win probability", ("57.9%", "58%", "0.579"))

    assert fact_found(home_win, answer)


@pytest.mark.parametrize("answer", ["Founded in 2015.", "A 1.5 goal average.", "15.5 shots"])
def test_fact_found_never_matches_a_number_inside_another_one(answer: str) -> None:
    assert not fact_found(Fact("points", ("15",)), answer)


@pytest.mark.parametrize("score", ["0-0", "1-0"])
def test_fact_found_never_takes_part_of_a_date_for_a_score(score: str) -> None:
    assert not fact_found(Fact("score", (score,)), "Played on 2026-10-01 and 2026-10-04.")


@pytest.mark.parametrize("score", ["0-0", "5-0"])
def test_fact_found_never_takes_part_of_a_kick_off_time_for_a_score(score: str) -> None:
    assert not fact_found(Fact("score", (score,)), "Kick-off at 20:00, then 15:00.")


def test_fact_found_ignores_case_and_accents() -> None:
    assert fact_found(Fact("leader", ("Saint-Étienne",)), "SAINT-ETIENNE leads the table.")


@pytest.mark.parametrize("answer", ["**Lens** lead.", "_Lens_ lead.", "- Lens: 13 points"])
def test_fact_found_sees_a_word_through_markdown(answer: str) -> None:
    assert fact_found(Fact("leader", ("Lens",)), answer)


def test_called_expected_tool_accepts_any_listed_tool_on_the_right_competition() -> None:
    calls = [ToolCall("get_matches", {"competition": "PL"}, "{}")]

    assert called_expected_tool(calls, ("get_team_form", "get_matches"), "PL")
    assert not called_expected_tool(calls, ("get_team_form", "get_matches"), "FL1")
    assert not called_expected_tool(calls, ("get_standings",), "PL")


def test_score_case_succeeds_only_with_the_right_tool_and_every_fact() -> None:
    calls = [
        ToolCall("get_standings", {"league": "FL1"}, "Error: unexpected keyword 'league'"),
        ToolCall("get_standings", {"competition": "FL1"}, "{}"),
    ]
    facts = [Fact("leader", ("Monaco",)), Fact("points", ("13",))]

    complete = score_case("Monaco lead with 13 points.", calls, ("get_standings",), "FL1", facts)
    partial = score_case("Monaco lead the table.", calls, ("get_standings",), "FL1", facts)

    assert complete == {
        "right_tool": True,
        "facts_found": 2,
        "facts_expected": 2,
        "failed_calls": 1,
        "success": True,
    }
    assert (partial["facts_found"], partial["success"]) == (1, False)
