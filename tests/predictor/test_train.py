from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.evaluation import OUTCOMES
from football_agent.predictor.features import FEATURE_COLUMNS
from football_agent.predictor.prepare import load_dataset
from football_agent.predictor.train import (
    EXPERIMENT_NAME,
    feature_importance,
    run_experiment,
    train_model,
)


def test_train_model_gives_one_probability_column_per_outcome(
    dataset_dir: Path, training_config: TrainingConfig
) -> None:
    _, parts = load_dataset(dataset_dir)

    model = train_model(parts["train"], parts["validation"], training_config)

    assert list(model.classes_) == list(range(len(OUTCOMES)))
    assert model.predict_proba(parts["validation"][FEATURE_COLUMNS]).shape == (
        len(parts["validation"]),
        3,
    )


def test_train_model_with_early_stopping_stops_before_the_iteration_limit(
    dataset_dir: Path, training_config: TrainingConfig
) -> None:
    _, parts = load_dataset(dataset_dir)
    config = training_config.model_copy(deep=True)
    config.model.n_estimators = 500
    config.training.early_stopping_rounds = 5

    model = train_model(parts["train"], parts["validation"], config)

    assert 0 < model.best_iteration_ < 500


def test_feature_importance_gives_split_and_gain_for_every_feature(
    dataset_dir: Path, training_config: TrainingConfig
) -> None:
    _, parts = load_dataset(dataset_dir)
    model = train_model(parts["train"], parts["validation"], training_config)

    importance = feature_importance(model)

    assert set(importance) == {"split", "gain"}
    for values in importance.values():
        assert list(values) == FEATURE_COLUMNS
        assert all(value >= 0 for value in values.values())
    assert sum(importance["split"].values()) > 0


def test_run_experiment_records_data_configs_metrics_and_model(
    dataset_dir: Path,
    training_config: TrainingConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MLflow writes its database and artifacts relative to the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    scores = run_experiment(dataset_dir, training_config)

    run = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list")[0]
    assert run.info.run_name == "test-run"
    assert run.data.metrics["validation_log_loss"] == pytest.approx(scores["log_loss"])
    assert {"naive_validation_log_loss", "market_validation_log_loss", "iterations"} <= set(
        run.data.metrics
    )
    assert (run.data.params["dataset"], run.data.params["train_seasons"]) == (
        "synthetic",
        "2016-2017",
    )
    assert sorted(item.dataset.name for item in run.inputs.dataset_inputs) == [
        "synthetic-training",
        "synthetic-validation",
    ]
    artifacts = run.info.artifact_uri
    training_yaml = mlflow.artifacts.load_text(f"{artifacts}/training_config.yaml")
    assert yaml.safe_load(training_yaml) == training_config.model_dump()
    assert set(mlflow.artifacts.load_dict(f"{artifacts}/feature_importance.json")) == {
        "split",
        "gain",
    }
    model = mlflow.lightgbm.load_model(f"runs:/{run.info.run_id}/model")
    assert list(model.classes_) == list(range(len(OUTCOMES)))


def test_run_experiment_refuses_a_dataset_without_the_requested_features(
    dataset_dir: Path, training_config: TrainingConfig
) -> None:
    train = pd.read_parquet(dataset_dir / "train.parquet").drop(columns="away_form_points")
    train.to_parquet(dataset_dir / "train.parquet", index=False)

    with pytest.raises(ValueError, match="away_form_points"):
        run_experiment(dataset_dir, training_config)
