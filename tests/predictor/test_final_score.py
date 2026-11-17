from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.evaluation import evaluate, naive_probabilities
from football_agent.predictor.final_score import TEST_METRICS, score_champion_on_test
from football_agent.predictor.prepare import load_dataset
from football_agent.predictor.registry import CHAMPION_ALIAS, MODEL_NAME, promote_run
from football_agent.predictor.train import run_experiment


@pytest.fixture
def champion_run_id(tracking: None, dataset_dir: Path, training_config: TrainingConfig) -> str:
    run_experiment(dataset_dir, training_config)
    return promote_run("test-run").run_id


def test_score_champion_on_test_records_the_scores_in_the_champion_run(
    champion_run_id: str, dataset_dir: Path
) -> None:
    scores = score_champion_on_test(dataset_dir.parent)

    metrics = MlflowClient().get_run(champion_run_id).data.metrics
    assert {name: metrics[metric] for name, metric in TEST_METRICS.items()} == pytest.approx(scores)
    assert metrics["cv_log_loss"] != scores["log_loss"]


def test_score_champion_on_test_gives_the_naive_baseline_every_season_before_the_test(
    champion_run_id: str, dataset_dir: Path
) -> None:
    _, parts = load_dataset(dataset_dir)
    before_test = pd.concat([parts["train"], parts["validation"]])
    naive = naive_probabilities(before_test, len(parts["test"]))

    scores = score_champion_on_test(dataset_dir.parent)

    assert scores["naive_log_loss"] == pytest.approx(
        evaluate(naive, parts["test"]["result"])["log_loss"]
    )


def test_score_champion_on_test_never_reads_the_test_seasons_twice(
    champion_run_id: str, dataset_dir: Path
) -> None:
    first = score_champion_on_test(dataset_dir.parent)
    (dataset_dir / "test.parquet").unlink()

    second = score_champion_on_test(dataset_dir.parent)

    assert second == pytest.approx(first)
    assert MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).run_id == (
        champion_run_id
    )
