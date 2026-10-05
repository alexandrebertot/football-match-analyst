from itertools import permutations
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.evaluation import OUTCOMES
from football_agent.predictor.features import FEATURE_COLUMNS, build_features
from football_agent.predictor.train import (
    EXPERIMENT_NAME,
    feature_importance,
    run_experiment,
    train_model,
)

BASELINE = TrainingConfig(run_name="test-run", features=FEATURE_COLUMNS, form_window=5)


def synthetic_league(seasons: range) -> pd.DataFrame:
    """Return a 6-team league where every team hosts every other once per season, random scores."""
    rng = np.random.default_rng(0)
    teams = ["PSG", "Lyon", "Nice", "Lens", "Lille", "Brest"]
    games = []
    for season in seasons:
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


def train_and_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_features(synthetic_league(range(2016, 2019)), window=5)
    return features[features["season"] < 2018], features[features["season"] == 2018]


def test_train_model_gives_one_probability_column_per_outcome() -> None:
    train, validation = train_and_validation()

    model = train_model(train, validation, BASELINE)

    assert list(model.classes_) == list(range(len(OUTCOMES)))
    assert model.predict_proba(validation[FEATURE_COLUMNS]).shape == (len(validation), 3)


def test_train_model_with_early_stopping_stops_before_the_iteration_limit() -> None:
    train, validation = train_and_validation()
    config = BASELINE.model_copy(
        update={"model": {"n_estimators": 500}, "early_stopping_rounds": 5}
    )

    model = train_model(train, validation, config)

    assert 0 < model.best_iteration_ < 500


def test_feature_importance_gives_split_and_gain_for_every_feature() -> None:
    model = train_model(*train_and_validation(), BASELINE)

    importance = feature_importance(model)

    assert set(importance) == {"split", "gain"}
    for values in importance.values():
        assert list(values) == FEATURE_COLUMNS
        assert all(value >= 0 for value in values.values())
    assert sum(importance["split"].values()) > 0


def test_run_experiment_records_params_metrics_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MLflow writes its database and artifacts relative to the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    scores = run_experiment(synthetic_league(range(2016, 2024)), BASELINE)

    run = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list")[0]
    assert run.data.metrics["validation_log_loss"] == pytest.approx(scores["log_loss"])
    assert {"naive_validation_log_loss", "market_validation_log_loss", "iterations"} <= set(
        run.data.metrics
    )
    assert run.data.params["features"] == ",".join(FEATURE_COLUMNS)
    model = mlflow.lightgbm.load_model(f"runs:/{run.info.run_id}/model")
    assert list(model.classes_) == list(range(len(OUTCOMES)))
    importance = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/feature_importance.json")
    assert set(importance) == {"split", "gain"}
    assert run.info.run_name == "test-run"
    logged_config = mlflow.artifacts.load_text(f"{run.info.artifact_uri}/config.yaml")
    assert yaml.safe_load(logged_config) == BASELINE.model_dump()
