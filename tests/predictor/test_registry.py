from pathlib import Path

import pytest
from mlflow import MlflowClient

from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.registry import (
    CHAMPION_ALIAS,
    MODEL_NAME,
    load_champion,
    promote_run,
)
from football_agent.predictor.train import run_experiment


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # MLflow writes its database and artifacts relative to the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_promote_run_registers_a_new_version_and_moves_the_champion_alias(
    tracking: None, dataset_dir: Path, training_config: TrainingConfig
) -> None:
    run_experiment(dataset_dir, training_config)
    first = promote_run("test-run")
    run_experiment(dataset_dir, training_config)

    second = promote_run("test-run")

    assert (int(first.version), int(second.version)) == (1, 2)
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert int(champion.version) == 2
    assert champion.run_id == second.run_id != first.run_id


def test_load_champion_returns_the_promoted_model(
    tracking: None, dataset_dir: Path, training_config: TrainingConfig
) -> None:
    run_experiment(dataset_dir, training_config)
    promote_run("test-run")

    model = load_champion()

    assert list(model.feature_name_) == training_config.features


def test_promote_run_refuses_an_unknown_run_name(tracking: None) -> None:
    with pytest.raises(ValueError, match="No run named 'missing'"):
        promote_run("missing")
