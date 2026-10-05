from pathlib import Path

import pytest
from pydantic import ValidationError

from football_agent.predictor.config import (
    SeasonRange,
    load_dataset_config,
    load_training_config,
)

CONFIGS_DIR = Path(__file__).parents[2] / "configs"

VALID_DATASET = """
name: form5
seasons:
  first: 2016
  last: 2025
split:
  train:
    first: 2016
    last: 2022
  validation:
    first: 2023
    last: 2023
  test:
    first: 2024
    last: 2025
features:
  form_window: 5
"""

VALID_TRAINING = """
run:
  name: baseline-form
dataset: form5
features:
  - home_form_points
  - away_form_points
model:
  n_estimators: 100
  learning_rate: 0.1
  num_leaves: 31
  max_depth: -1
  min_child_samples: 20
  subsample: 1.0
  subsample_freq: 0
  colsample_bytree: 1.0
  reg_alpha: 0.0
  reg_lambda: 0.0
training:
  early_stopping_rounds: null
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_season_range_includes_both_ends() -> None:
    assert SeasonRange(first=2016, last=2022).to_range() == range(2016, 2023)


def test_load_dataset_config_reads_a_valid_file(tmp_path: Path) -> None:
    config = load_dataset_config(write_config(tmp_path, VALID_DATASET))

    assert config.name == "form5"
    assert config.split.validation.to_range() == range(2023, 2024)
    assert config.features.form_window == 5


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  form_window: 5", "  form_windw: 5"),
        ("  form_window: 5", "  form_window: 0"),
        ("    last: 2022\n", "    last: 2023\n"),
        ("    last: 2022\n", "    last: 2015\n"),
        ("  last: 2025\nsplit", "  last: 2024\nsplit"),
    ],
    ids=[
        "unknown-key",
        "non-positive-window",
        "train-overlaps-validation",
        "season-range-reversed",
        "split-outside-loaded-seasons",
    ],
)
def test_load_dataset_config_rejects_invalid_files(tmp_path: Path, old: str, new: str) -> None:
    assert old in VALID_DATASET
    with pytest.raises(ValidationError):
        load_dataset_config(write_config(tmp_path, VALID_DATASET.replace(old, new)))


def test_load_training_config_reads_a_valid_file(tmp_path: Path) -> None:
    config = load_training_config(write_config(tmp_path, VALID_TRAINING))

    assert config.run.name == "baseline-form"
    assert config.model.learning_rate == 0.1
    assert config.training.early_stopping_rounds is None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  num_leaves: 31\n", ""),
        ("  learning_rate: 0.1", "  learnig_rate: 0.1"),
        ("  learning_rate: 0.1", "  learning_rate: 0"),
        ("  - away_form_points", "  - away_form_pts"),
        ("early_stopping_rounds: null", "early_stopping_rounds: 0"),
    ],
    ids=[
        "missing-hyperparameter",
        "misspelled-hyperparameter",
        "non-positive-learning-rate",
        "unknown-feature",
        "non-positive-early-stopping",
    ],
)
def test_load_training_config_rejects_invalid_files(tmp_path: Path, old: str, new: str) -> None:
    assert old in VALID_TRAINING
    with pytest.raises(ValidationError):
        load_training_config(write_config(tmp_path, VALID_TRAINING.replace(old, new)))


def test_repository_configs_are_valid_and_training_uses_existing_datasets() -> None:
    dataset_names = {
        load_dataset_config(path).name for path in (CONFIGS_DIR / "datasets").glob("*.yaml")
    }
    training_paths = list((CONFIGS_DIR / "training").glob("*.yaml"))

    assert dataset_names
    assert training_paths
    for path in training_paths:
        assert load_training_config(path).dataset in dataset_names, path.name
