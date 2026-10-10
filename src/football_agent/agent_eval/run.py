import argparse
import json
import math
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai.scorers import scorer

from football_agent.agent import answer
from football_agent.agent_eval.cases import ACCEPTED_TOOLS, EvalCase, build_cases, expected_facts
from football_agent.agent_eval.scoring import Fact, ToolCall, fact_found, score_case
from football_agent.agent_eval.snapshot import SnapshotClient
from football_agent.llm import MODEL, make_llm_client
from football_agent.predictor.data import RAW_DATA_DIR
from football_agent.predictor.predict import load_predictor
from football_agent.tools import ToolContext

EVALUATION_EXPERIMENT = "agent-evaluation"
# 95% of a normal distribution lies within 1.96 standard deviations of its mean.
Z_95 = 1.96


def evaluation_data(
    cases: list[EvalCase], facts: list[list[Fact]], passes: int
) -> list[dict[str, Any]]:
    """One row per question and pass, in the format of mlflow.genai.evaluate."""
    return [
        {
            "inputs": {"question": case.question},
            "expectations": {
                "tools": list(ACCEPTED_TOOLS[case.kind]),
                "competition": case.competition,
                "facts": [
                    {"description": fact.description, "spellings": list(fact.spellings)}
                    for fact in case_facts
                ],
            },
            "tags": {"kind": case.kind, "competition": case.competition, "pass": str(number)},
        }
        for number in range(passes)
        for case, case_facts in zip(cases, facts, strict=True)
    ]


def tool_calls_from_trace(trace: Trace) -> list[ToolCall]:
    """Read the tool calls of one answer back from its trace, in the order they were made."""
    spans = sorted(trace.search_spans(span_type=SpanType.TOOL), key=lambda span: span.start_time_ns)
    return [
        ToolCall(
            span.name,
            parse_arguments(span.inputs["arguments"]),
            # A tool that raised an unexpected exception has no result: the agent stopped there.
            span.outputs["result"] if span.outputs else "Error: the tool raised an exception.",
        )
        for span in spans
    ]


def parse_arguments(text: str) -> dict[str, Any]:
    # Arguments that are not a JSON object already gave the agent an error result, which the
    # scoring counts.
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


@scorer
def answer_score(outputs: str | None, expectations: dict[str, Any], trace: Trace) -> list[Feedback]:
    # None when the agent raised (cut-off or empty reply, too many rounds): the answer fails.
    text = outputs or ""
    facts = [Fact(fact["description"], tuple(fact["spellings"])) for fact in expectations["facts"]]
    score = score_case(
        text,
        tool_calls_from_trace(trace),
        tuple(expectations["tools"]),
        expectations["competition"],
        facts,
    )
    missing = [fact.description for fact in facts if not fact_found(fact, text)]
    return [
        Feedback(name="success", value=score["success"]),
        Feedback(name="right_tool", value=score["right_tool"]),
        Feedback(
            name="facts_found",
            value=score["facts_found"] / score["facts_expected"],
            rationale=f"Missing: {', '.join(missing)}" if missing else "Every fact found.",
        ),
        Feedback(name="failed_calls", value=score["failed_calls"]),
    ]


def success_interval(results: pd.DataFrame) -> tuple[float, float]:
    """Success rate and the half-width of its 95% confidence interval.

    The passes of one question are not independent (a hard question tends to fail every time),
    so the interval counts questions, not answers.
    """
    # MLflow records a scorer that raised as a missing value, which the mean would skip.
    unscored = results["success/value"].isna().sum()
    if unscored:
        raise ValueError(f"{unscored} answers could not be scored: see the errors in their traces.")
    questions = results["request"].map(lambda request: request["question"])
    per_question = results["success/value"].astype(float).groupby(questions).mean()
    return per_question.mean(), Z_95 * per_question.std() / math.sqrt(len(per_question))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the agent on a recorded snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()

    # Ollama answers one question at a time on a single GPU; and without the second setting,
    # MLflow asks the first question once more to check that the agent emits a trace.
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    today = date.fromisoformat(snapshot["date"])
    predictor = load_predictor(RAW_DATA_DIR, today)
    # The predictor must only know the matches played before the snapshot, so that every
    # evaluation of this snapshot asks for the same predictions.
    played = predictor.history[predictor.history["date"] < pd.Timestamp(today)]
    context = ToolContext(SnapshotClient(snapshot), replace(predictor, history=played), today)
    cases = build_cases(snapshot)
    facts = [expected_facts(case, context) for case in cases]
    llm = make_llm_client()
    # MLflow records an answer that raised as a failure and goes on: without this check, an
    # Ollama that is not running would give a success rate of 0%.
    llm.models.list()

    mlflow.set_experiment(EVALUATION_EXPERIMENT)
    mlflow.openai.autolog()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": MODEL,
                "snapshot": snapshot["date"],
                "questions": len(cases),
                "passes": args.passes,
            }
        )
        result = mlflow.genai.evaluate(
            data=evaluation_data(cases, facts, args.passes),
            predict_fn=lambda question: answer(question, llm, context),
            scorers=[answer_score],
        )
        rate, margin = success_interval(result.result_df)
        # Answers where the agent raised: its own errors, or Ollama stopping during the run.
        errors = int((result.result_df["state"] == "ERROR").sum())
        mlflow.log_metrics(
            {"success_rate": rate, "success_margin_95": margin, "answers_with_errors": errors}
        )
    print(f"Success: {rate:.1%} ± {margin:.1%} (95% confidence interval)")
    print(f"Answers where the agent raised an error: {errors}, see their traces.")
