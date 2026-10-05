import argparse
from pathlib import Path
from typing import Any

import lightgbm
import mlflow
import mlflow.data
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier

from football_agent.predictor.config import TrainingConfig, load_training_config
from football_agent.predictor.evaluation import (
    OUTCOMES,
    evaluate,
    market_probabilities,
    naive_probabilities,
)
from football_agent.predictor.prepare import PROCESSED_DATA_DIR, load_dataset

EXPERIMENT_NAME = "match-outcome"


def train_model(
    train: pd.DataFrame, validation: pd.DataFrame, config: TrainingConfig
) -> LGBMClassifier:
    """Train on `train`; with early stopping, stop when the validation log-loss stops improving."""
    # Labels are encoded as OUTCOMES indices, so probability column i is OUTCOMES[i]. LightGBM's
    # eval_y does not accept text labels.
    codes = {outcome: code for code, outcome in enumerate(OUTCOMES)}
    model = LGBMClassifier(random_state=0, verbose=-1, **config.model.model_dump())
    fit_options: dict[str, Any] = {}
    if config.training.early_stopping_rounds is not None:
        fit_options = {
            "eval_X": (validation[config.features],),
            "eval_y": (validation["result"].map(codes),),
            "callbacks": [
                lightgbm.early_stopping(config.training.early_stopping_rounds, verbose=False)
            ],
        }
    model.fit(train[config.features], train["result"].map(codes), **fit_options)
    if len(model.classes_) != len(OUTCOMES):
        raise ValueError(f"Training data covers only classes {list(model.classes_)}.")
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


def run_experiment(dataset_dir: Path, config: TrainingConfig) -> dict[str, float]:
    """Train on a prepared dataset, score its validation part, and record everything in MLflow."""
    dataset_config, parts = load_dataset(dataset_dir)
    train, validation = parts["train"], parts["validation"]
    missing = set(config.features) - set(train.columns)
    if missing:
        raise ValueError(
            f"Dataset {dataset_config.name} has no {sorted(missing)}: prepare it again"
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=config.run.name):
        # Links the run to the exact data it used: the digest changes whenever the data does.
        for context, rows in [("training", train), ("validation", validation)]:
            name = f"{dataset_config.name}-{context}"
            mlflow.log_input(mlflow.data.from_pandas(rows, name=name), context=context)
        model = train_model(train, validation, config)
        scores = evaluate(model.predict_proba(validation[config.features]), validation["result"])
        naive = evaluate(naive_probabilities(train, len(validation)), validation["result"])
        market = evaluate(market_probabilities(validation), validation["result"])
        split = dataset_config.split
        mlflow.log_params(
            {
                "dataset": dataset_config.name,
                "features": ",".join(config.features),
                "form_window": dataset_config.features.form_window,
                "train_seasons": f"{split.train.first}-{split.train.last}",
                "validation_seasons": f"{split.validation.first}-{split.validation.last}",
                "early_stopping_rounds": config.training.early_stopping_rounds,
                **model.get_params(),
            }
        )
        mlflow.log_metrics(
            {
                "validation_log_loss": scores["log_loss"],
                "validation_accuracy": scores["accuracy"],
                "naive_validation_log_loss": naive["log_loss"],
                "market_validation_log_loss": market["log_loss"],
                "iterations": model.best_iteration_ or model.n_estimators_,
            }
        )
        mlflow.log_dict(config.model_dump(), "training_config.yaml")
        mlflow.log_dict(dataset_config.model_dump(), "dataset_config.yaml")
        mlflow.log_dict(feature_importance(model), "feature_importance.json")
        mlflow.lightgbm.log_model(model, name="model")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a match outcome model from a config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    config = load_training_config(parser.parse_args().config)
    print(run_experiment(PROCESSED_DATA_DIR / config.dataset, config))
