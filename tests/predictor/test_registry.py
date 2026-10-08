from pathlib import Path

import pytest
from mlflow import MlflowClient

from football_agent.predictor.config import DatasetConfig, TrainingConfig
from football_agent.predictor.registry import (
    CHAMPION_ALIAS,
    MODEL_NAME,
    load_champion,
    promote_run,
)
from football_agent.predictor.train import run_experiment


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


def test_load_champion_returns_the_promoted_model_and_its_dataset_config(
    tracking: None,
    dataset_dir: Path,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
) -> None:
    run_experiment(dataset_dir, training_config)
    promote_run("test-run")

    model, champion_dataset = load_champion()

    assert list(model.feature_name_) == training_config.features
    assert champion_dataset == dataset_config


def test_promote_run_refuses_an_unknown_run_name(tracking: None) -> None:
    with pytest.raises(ValueError, match="No run named 'missing'"):
        promote_run("missing")
