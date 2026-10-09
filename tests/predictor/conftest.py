from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_agent.predictor.config import DatasetConfig, TrainingConfig
from football_agent.predictor.features import FEATURE_COLUMNS
from football_agent.predictor.prepare import prepare_dataset


@pytest.fixture
def league() -> pd.DataFrame:
    """A 6-team league over 2016-17 to 2019-20: every team hosts every other once, random scores."""
    rng = np.random.default_rng(0)
    teams = ["PSG", "Lyon", "Nice", "Lens", "Lille", "Brest"]
    games = []
    for season in range(2016, 2020):
        for day, (home, away) in enumerate(permutations(teams, 2)):
            home_goals, away_goals = rng.integers(0, 4, 2)
            games.append(
                {
                    "date": pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=7 * day),
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_shots_on_target": home_goals + rng.integers(0, 5),
                    "away_shots_on_target": away_goals + rng.integers(0, 5),
                    "average_odds_home": 2.5,
                    "average_odds_draw": 3.2,
                    "average_odds_away": 2.9,
                }
            )
    matches = pd.DataFrame(games)
    matches["result"] = np.select(
        [
            matches["home_goals"] > matches["away_goals"],
            matches["home_goals"] < matches["away_goals"],
        ],
        ["H", "A"],
        "D",
    )
    return matches


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "name": "synthetic",
            "seasons": {"first": 2016, "last": 2019},
            "split": {
                "train": {"first": 2016, "last": 2017},
                "validation": {"first": 2018, "last": 2018},
                "test": {"first": 2019, "last": 2019},
            },
            "features": {"form_window": 5},
        }
    )


@pytest.fixture
def training_config() -> TrainingConfig:
    """LightGBM defaults without early stopping, like configs/training/baseline.yaml."""
    return TrainingConfig.model_validate(
        {
            "run": {"name": "test-run"},
            "dataset": "synthetic",
            "features": FEATURE_COLUMNS,
            "model": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 1.0,
                "subsample_freq": 0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
            },
            "training": {"early_stopping_rounds": None},
        }
    )


@pytest.fixture
def dataset_dir(tmp_path: Path, league: pd.DataFrame, dataset_config: DatasetConfig) -> Path:
    return prepare_dataset(league, dataset_config, tmp_path / "processed")
