from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from football_agent.predictor.compare import compare_runs, compare_scores
from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.registry import latest_run
from football_agent.predictor.train import run_experiment


def match_scores(log_losses: list[float], seasons: list[int] | None = None) -> pd.DataFrame:
    """Scores of matches between the same teams, one per day, with the given log-losses."""
    seasons = seasons or [2021] * len(log_losses)
    return pd.DataFrame(
        {
            "season": seasons,
            "date": [f"2021-09-{day + 1:02d}" for day in range(len(log_losses))],
            "home_team": "PSG",
            "away_team": "Lyon",
            "result": "H",
            "log_loss": log_losses,
        }
    )


def test_compare_scores_keeps_a_candidate_better_on_every_match() -> None:
    baseline = match_scores([0.4, 1.2, 0.9, 1.6])
    candidate = match_scores([0.39, 1.18, 0.89, 1.59])

    comparison = compare_scores(baseline, candidate)

    assert comparison["difference"] == pytest.approx(-0.0125)
    assert comparison["keep_candidate"]


def test_compare_scores_rejects_a_candidate_whose_interval_reaches_zero() -> None:
    # Better by 0.01 on average, but by +0.2 or -0.22 depending on the match.
    baseline = match_scores([1.0, 1.0, 1.0, 1.0])
    candidate = match_scores([1.2, 0.78, 1.2, 0.78])

    comparison = compare_scores(baseline, candidate)

    assert comparison["difference"] == pytest.approx(-0.01)
    assert comparison["difference"] + comparison["half_width"] > 0
    assert not comparison["keep_candidate"]


def test_compare_scores_gives_the_mean_difference_of_each_season() -> None:
    baseline = match_scores([1.0, 1.0, 1.0], seasons=[2021, 2021, 2022])
    candidate = match_scores([0.9, 0.8, 1.1], seasons=[2021, 2021, 2022])

    comparison = compare_scores(baseline, candidate)

    assert comparison["per_season"] == pytest.approx({2021: -0.15, 2022: 0.1})


def test_compare_scores_refuses_runs_that_scored_different_matches() -> None:
    baseline = match_scores([1.0, 1.0])
    candidate = match_scores([1.0, 1.0]).assign(away_team=["Lyon", "Nice"])

    with pytest.raises(ValueError, match="same matches"):
        compare_scores(baseline, candidate)


def test_compare_runs_records_the_comparison_and_decision_in_the_candidate_run(
    tracking: None, dataset_dir: Path, training_config: TrainingConfig
) -> None:
    candidate_config = training_config.model_copy(deep=True)
    candidate_config.run.name = "candidate"
    candidate_config.features = ["home_points_avg", "away_points_avg"]
    run_experiment(dataset_dir, training_config)
    run_experiment(dataset_dir, candidate_config)

    comparison = compare_runs("test-run", "candidate")

    run = MlflowClient().get_run(latest_run("candidate").info.run_id)
    assert run.data.metrics["paired_difference"] == pytest.approx(comparison["difference"])
    assert run.data.metrics["paired_half_width"] == pytest.approx(comparison["half_width"])
    assert run.data.tags["compared_with"].startswith("test-run (")
    assert run.data.tags["decision"] == ("keep" if comparison["keep_candidate"] else "reject")
