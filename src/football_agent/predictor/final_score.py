from pathlib import Path

import mlflow
from mlflow import MlflowClient

from football_agent.predictor.evaluation import compare_with_baselines, gap_closed
from football_agent.predictor.prepare import PROCESSED_DATA_DIR, load_dataset
from football_agent.predictor.registry import CHAMPION_ALIAS, MODEL_NAME, load_champion

# Same naming as the validation metrics that train.py logs.
TEST_METRICS = {
    "log_loss": "test_log_loss",
    "accuracy": "test_accuracy",
    "naive_log_loss": "naive_test_log_loss",
    "market_log_loss": "market_test_log_loss",
}


def score_champion_on_test(processed_dir: Path) -> dict[str, float]:
    """Score the champion on the held-out test seasons of its dataset, and record it in its run.

    The test seasons are looked at only once: if the champion's run already holds a test score,
    that score is returned and the test seasons are not read again.
    """
    client = MlflowClient()
    run_id = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).run_id
    recorded = client.get_run(run_id).data.metrics
    if TEST_METRICS["log_loss"] in recorded:
        return {name: recorded[metric] for name, metric in TEST_METRICS.items()}

    model, dataset_config = load_champion()
    _, parts = load_dataset(processed_dir / dataset_config.name)
    test = parts["test"]
    probabilities = model.predict_proba(test[model.feature_name_])
    scores = compare_with_baselines(probabilities, parts["train"], test)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({metric: scores[name] for name, metric in TEST_METRICS.items()})
    return scores


if __name__ == "__main__":
    scores = score_champion_on_test(PROCESSED_DATA_DIR)
    print({**scores, "gap_closed": gap_closed(scores)})
