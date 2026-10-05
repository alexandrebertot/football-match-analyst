import argparse
from pathlib import Path

import pandas as pd
import yaml

from football_agent.predictor.config import DatasetConfig, load_dataset_config, read_yaml
from football_agent.predictor.data import RAW_DATA_DIR, load_matches
from football_agent.predictor.evaluation import temporal_split
from football_agent.predictor.features import build_features

PROCESSED_DATA_DIR = Path("data/processed")
PARTS = ["train", "validation", "test"]


def prepare_dataset(matches: pd.DataFrame, config: DatasetConfig, output_dir: Path) -> Path:
    """Build the features, split them and save each part, along with the config that made them."""
    features = build_features(matches, config.features.form_window)
    dataset_dir = output_dir / config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for part, rows in zip(PARTS, temporal_split(features, config.split), strict=True):
        rows.to_parquet(dataset_dir / f"{part}.parquet", index=False)
    (dataset_dir / "dataset.yaml").write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
    return dataset_dir


def load_dataset(dataset_dir: Path) -> tuple[DatasetConfig, dict[str, pd.DataFrame]]:
    """Return the config a dataset was prepared with and its train, validation and test parts."""
    config = DatasetConfig.model_validate(read_yaml(dataset_dir / "dataset.yaml"))
    parts = {part: pd.read_parquet(dataset_dir / f"{part}.parquet") for part in PARTS}
    return config, parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a feature dataset from a config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML dataset config")
    config = load_dataset_config(parser.parse_args().config)
    matches = load_matches(RAW_DATA_DIR, config.seasons.to_range())
    print(f"Dataset written to {prepare_dataset(matches, config, PROCESSED_DATA_DIR)}")
