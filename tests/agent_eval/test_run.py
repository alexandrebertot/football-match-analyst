import math
import statistics
from datetime import date

import mlflow
import pandas as pd
import pytest

from fakes import FakeLLM, text_reply, tool_reply
from football_agent.agent import answer
from football_agent.agent_eval.cases import EvalCase
from football_agent.agent_eval.run import (
    answer_score,
    evaluation_data,
    parse_arguments,
    success_interval,
)
from football_agent.agent_eval.scoring import Fact
from football_agent.agent_eval.snapshot import SnapshotClient
from football_agent.tools import ToolContext

LEADER = EvalCase("leader", "Who is top of the Ligue 1?", "FL1", {"competition": "FL1"})
LEADER_FACTS = [Fact("leader", ("Nice", "OGC Nice"))]
SNAPSHOT = {
    "date": "2026-10-09",
    "competitions": {
        "FL1": {
            "standings": {
                "competition": {"name": "Ligue 1"},
                "season": {"startDate": "2026-08-14"},
                "standings": [
                    {
                        "table": [
                            {
                                "position": 1,
                                "team": {"id": 522, "name": "OGC Nice", "shortName": "Nice"},
                                "playedGames": 7,
                                "won": 5,
                                "draw": 1,
                                "lost": 1,
                                "goalsFor": 12,
                                "goalsAgainst": 5,
                                "goalDifference": 7,
                                "points": 16,
                            }
                        ]
                    }
                ],
            },
            "teams": {
                "teams": [
                    {"id": 522, "name": "OGC Nice", "shortName": "Nice"},
                    {"id": 546, "name": "Racing Club de Lens", "shortName": "Lens"},
                ]
            },
            "matches": {"matches": []},
        }
    },
}


def test_evaluation_data_asks_every_question_once_per_pass() -> None:
    position = EvalCase("position", "Where are Lens?", "FL1", {"competition": "FL1"}, "Lens")
    facts = [LEADER_FACTS, [Fact("position 2", ("2nd", "second"))]]

    rows = evaluation_data([LEADER, position], facts, passes=2)

    assert [(row["inputs"]["question"], row["tags"]["pass"]) for row in rows] == [
        ("Who is top of the Ligue 1?", "0"),
        ("Where are Lens?", "0"),
        ("Who is top of the Ligue 1?", "1"),
        ("Where are Lens?", "1"),
    ]
    assert rows[1]["expectations"] == {
        "tools": ["get_standings"],
        "competition": "FL1",
        "facts": [{"description": "position 2", "spellings": ["2nd", "second"]}],
    }
    assert rows[1]["tags"] == {"kind": "position", "competition": "FL1", "pass": "0"}


def test_evaluation_scores_each_answer_from_its_trace(
    tracing: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One worker, so that the scripted replies are consumed in the order of the passes.
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")
    llm = FakeLLM(
        [
            # Pass 0: a broken call, fixed, then a right answer.
            tool_reply("get_standings", '{"competition": '),
            tool_reply("get_standings", '{"competition": "FL1"}'),
            text_reply("**Nice** are top with 16 points."),
            # Pass 1: an answer from memory, wrong.
            text_reply("Lens are top."),
            # Pass 2: an empty reply, which makes the agent raise.
            text_reply(""),
            # Pass 3: a tool that crashes (there is no predictor here), which stops the agent.
            tool_reply(
                "predict_match", '{"home_team": "Nice", "away_team": "Lens", "competition": "FL1"}'
            ),
        ]
    )
    context = ToolContext(SnapshotClient(SNAPSHOT), None, date(2026, 10, 9))

    result = mlflow.genai.evaluate(
        data=evaluation_data([LEADER], [LEADER_FACTS], passes=4),
        predict_fn=lambda question: answer(question, llm, context),
        scorers=[answer_score],
    )

    rows = result.result_df.sort_values("request_time")
    scores = rows[["success/value", "right_tool/value", "facts_found/value", "failed_calls/value"]]
    assert scores.values.tolist() == [
        [True, True, 1.0, 1],
        [False, False, 0.0, 0],
        [False, False, 0.0, 0],
        [False, False, 0.0, 1],
    ]
    assert rows["state"].tolist() == ["OK", "OK", "ERROR", "ERROR"]
    wrong_answer = {note["assessment_name"]: note for note in rows["assessments"].iloc[1]}
    assert wrong_answer["facts_found"]["rationale"] == "Missing: leader"


def test_success_interval_counts_questions_not_answers() -> None:
    results = pd.DataFrame(
        {
            "request": [{"question": question} for question in ["A", "B", "C"] * 3],
            "success/value": [True, True, False, True, False, False, True, False, False],
        }
    )

    rate, margin = success_interval(results)

    per_question = [1, 1 / 3, 0]
    assert rate == pytest.approx(4 / 9)
    assert margin == pytest.approx(1.96 * statistics.stdev(per_question) / math.sqrt(3))


def test_success_interval_refuses_answers_that_could_not_be_scored() -> None:
    results = pd.DataFrame(
        {"request": [{"question": "A"}, {"question": "B"}], "success/value": [True, None]}
    )

    with pytest.raises(ValueError, match="1 answers could not be scored"):
        success_interval(results)


@pytest.mark.parametrize(
    ("text", "expected"),
    [('{"competition": "PL"}', {"competition": "PL"}), ('{"competition": ', {}), ('["PL"]', {})],
)
def test_parse_arguments_keeps_only_a_json_object(text: str, expected: dict[str, str]) -> None:
    assert parse_arguments(text) == expected
