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
    market_probabilities,
    match_log_losses,
    naive_probabilities,
    summarize_folds,
)
from football_agent.predictor.prepare import PROCESSED_DATA_DIR, load_dataset

EXPERIMENT_NAME = "match-outcome"
# What identifies a scored match, so that two runs can be compared match by match.
MATCH_COLUMNS = ["season", "date", "home_team", "away_team", "result"]
MATCH_SCORES_FILE = "match_scores.csv"


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


def fold_parts(
    rows: pd.DataFrame, season: int, early_stopping: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `rows` for the fold that scores `season`: the matches to fit on, the matches that
    stop the training, and the matches scored.

    With early stopping, the season just before `season` stops the training, so that the scored
    season never chooses the number of trees.
    """
    scored = rows[rows["season"] == season]
    if not early_stopping:
        return rows[rows["season"] < season], rows.iloc[:0], scored
    return rows[rows["season"] < season - 1], rows[rows["season"] == season - 1], scored


def cross_validate(
    rows: pd.DataFrame, validation_seasons: range, config: TrainingConfig
) -> pd.DataFrame:
    """Score every match of `validation_seasons`, each season with a model trained on the seasons
    before it, along with the naive and market baselines.

    Returns one row per scored match: its season, date, teams and result, then the log-loss of the
    model, of the naive baseline and of the market.
    """
    early_stopping = config.training.early_stopping_rounds is not None
    folds = []
    for season in validation_seasons:
        fit, stopping, scored = fold_parts(rows, season, early_stopping)
        model = train_model(fit, stopping, config)
        results = scored["result"]
        folds.append(
            scored[MATCH_COLUMNS].assign(
                log_loss=match_log_losses(model.predict_proba(scored[config.features]), results),
                naive_log_loss=match_log_losses(naive_probabilities(fit, len(scored)), results),
                market_log_loss=match_log_losses(market_probabilities(scored), results),
            )
        )
    return pd.concat(folds, ignore_index=True)


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
    """Cross-validate a config on the validation seasons of a prepared dataset, train the model to
    keep on the seasons before the test ones, and record everything in MLflow."""
    dataset_config, parts = load_dataset(dataset_dir)
    train, validation = parts["train"], parts["validation"]
    missing = set(config.features) - set(train.columns)
    if missing:
        raise ValueError(
            f"Dataset {dataset_config.name} has no {sorted(missing)}: prepare it again"
        )
    before_test = pd.concat([train, validation], ignore_index=True)
    split = dataset_config.split

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=config.run.name):
        # Links the run to the exact data it used: the digest changes whenever the data does.
        for context, rows in [("training", train), ("validation", validation)]:
            name = f"{dataset_config.name}-{context}"
            mlflow.log_input(mlflow.data.from_pandas(rows, name=name), context=context)
        scores = cross_validate(before_test, split.validation.to_range(), config)
        metrics = summarize_folds(scores)
        # The model to keep is trained like one more fold, the one that would score the first
        # test season, so that the cross-validation measures the recipe that makes it.
        early_stopping = config.training.early_stopping_rounds is not None
        fit, stopping, _ = fold_parts(before_test, split.test.first, early_stopping)
        model = train_model(fit, stopping, config)
        mlflow.log_params(
            {
                "dataset": dataset_config.name,
                "features": ",".join(config.features),
                "form_window": dataset_config.features.form_window,
                "train_seasons": f"{split.train.first}-{split.train.last}",
                "validation_seasons": f"{split.validation.first}-{split.validation.last}",
                "fit_seasons": f"{fit['season'].min()}-{fit['season'].max()}",
                "early_stopping_rounds": config.training.early_stopping_rounds,
                **model.get_params(),
            }
        )
        mlflow.log_metrics({**metrics, "iterations": model.best_iteration_ or model.n_estimators_})
        mlflow.log_text(scores.to_csv(index=False), MATCH_SCORES_FILE)
        mlflow.log_dict(config.model_dump(), "training_config.yaml")
        mlflow.log_dict(dataset_config.model_dump(), "dataset_config.yaml")
        mlflow.log_dict(feature_importance(model), "feature_importance.json")
        mlflow.lightgbm.log_model(model, name="model")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a match outcome model from a config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    config = load_training_config(parser.parse_args().config)
    print(run_experiment(PROCESSED_DATA_DIR / config.dataset, config))
