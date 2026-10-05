import argparse

import mlflow
import mlflow.lightgbm
import yaml
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from football_agent.predictor.config import DatasetConfig
from football_agent.predictor.train import EXPERIMENT_NAME

MODEL_NAME = "match-outcome"
CHAMPION_ALIAS = "champion"


def promote_run(run_name: str) -> ModelVersion:
    """Register the model of the latest run named `run_name` and make it the champion."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError(f"No run named '{run_name}' in experiment '{EXPERIMENT_NAME}'.")
    # MLflow 3 stores a logged model as its own entity, linked to the run that produced it.
    model_id = runs[0].outputs.model_outputs[0].model_id
    version = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    return version


def load_champion() -> tuple[LGBMClassifier, DatasetConfig]:
    """Return the champion model and the config of the dataset it was trained on.

    Predictions must build features exactly like that dataset did (same form window).
    """
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    dataset_yaml = mlflow.artifacts.load_text(f"runs:/{version.run_id}/dataset_config.yaml")
    return model, DatasetConfig.model_validate(yaml.safe_load(dataset_yaml))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make the model of a run the champion.")
    parser.add_argument("--run", required=True, help="name of the training run to promote")
    version = promote_run(parser.parse_args().run)
    print(f"{MODEL_NAME} version {version.version} is now '{CHAMPION_ALIAS}'.")
