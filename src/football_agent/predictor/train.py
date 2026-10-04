import argparse
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier

from football_agent.predictor.config import TrainingConfig, load_config
from football_agent.predictor.data import RAW_DATA_DIR, SEASONS, load_matches
from football_agent.predictor.evaluation import (
    OUTCOMES,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    evaluate,
    market_probabilities,
    naive_probabilities,
    temporal_split,
)
from football_agent.predictor.features import build_features

EXPERIMENT_NAME = "match-outcome"


def train_model(train: pd.DataFrame, features: list[str], params: dict[str, Any]) -> LGBMClassifier:
    model = LGBMClassifier(random_state=0, verbose=-1, **params)
    model.fit(train[features], train["result"])
    # predict_proba columns follow model.classes_; every metric assumes the OUTCOMES order.
    if list(model.classes_) != OUTCOMES:
        raise ValueError(f"Unexpected class order {list(model.classes_)}, expected {OUTCOMES}.")
    return model


def feature_importance(model: LGBMClassifier) -> dict[str, dict[str, float]]:
    """Return how often each feature is used to split ("split") and how much it reduces the loss
    ("gain")."""
    return {
        importance_type: dict(
            zip(
                model.feature_name_,
                model.booster_.feature_importance(importance_type=importance_type).tolist(),
                strict=True,
            )
        )
        for importance_type in ["split", "gain"]
    }


def run_experiment(matches: pd.DataFrame, config: TrainingConfig) -> dict[str, float]:
    """Train on the train seasons, score on validation, and record everything in MLflow."""
    train, validation, _ = temporal_split(build_features(matches, config.form_window))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=config.run_name):
        model = train_model(train, config.features, config.model)
        probabilities = model.predict_proba(validation[config.features])
        scores = evaluate(probabilities, validation["result"])
        naive = evaluate(naive_probabilities(train, len(validation)), validation["result"])
        market = evaluate(market_probabilities(validation), validation["result"])
        mlflow.log_params(
            {
                "features": ",".join(config.features),
                "form_window": config.form_window,
                "train_seasons": f"{TRAIN_SEASONS.start}-{TRAIN_SEASONS.stop - 1}",
                "validation_seasons": f"{VALIDATION_SEASONS.start}-{VALIDATION_SEASONS.stop - 1}",
                **model.get_params(),
            }
        )
        mlflow.log_metrics(
            {
                "validation_log_loss": scores["log_loss"],
                "validation_accuracy": scores["accuracy"],
                "naive_validation_log_loss": naive["log_loss"],
                "market_validation_log_loss": market["log_loss"],
            }
        )
        mlflow.log_dict(config.model_dump(), "config.yaml")
        mlflow.log_dict(feature_importance(model), "feature_importance.json")
        mlflow.lightgbm.log_model(model, name="model")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a match outcome model from a config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    args = parser.parse_args()
    print(run_experiment(load_matches(RAW_DATA_DIR, SEASONS), load_config(args.config)))
