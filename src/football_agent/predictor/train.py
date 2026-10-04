import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier

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
from football_agent.predictor.features import FEATURE_COLUMNS, FORM_WINDOW, build_features

EXPERIMENT_NAME = "match-outcome"


def train_model(train: pd.DataFrame) -> LGBMClassifier:
    model = LGBMClassifier(random_state=0, verbose=-1)
    model.fit(train[FEATURE_COLUMNS], train["result"])
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
                FEATURE_COLUMNS,
                model.booster_.feature_importance(importance_type=importance_type).tolist(),
                strict=True,
            )
        )
        for importance_type in ["split", "gain"]
    }


def run_experiment(matches: pd.DataFrame) -> dict[str, float]:
    """Train on the train seasons, score on validation, and record everything in MLflow."""
    train, validation, _ = temporal_split(build_features(matches))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        model = train_model(train)
        scores = evaluate(model.predict_proba(validation[FEATURE_COLUMNS]), validation["result"])
        naive = evaluate(naive_probabilities(train, len(validation)), validation["result"])
        market = evaluate(market_probabilities(validation), validation["result"])
        mlflow.log_params(
            {
                "features": ",".join(FEATURE_COLUMNS),
                "form_window": FORM_WINDOW,
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
        mlflow.log_dict(feature_importance(model), "feature_importance.json")
        mlflow.lightgbm.log_model(model, name="model")
    return scores


if __name__ == "__main__":
    print(run_experiment(load_matches(RAW_DATA_DIR, SEASONS)))
