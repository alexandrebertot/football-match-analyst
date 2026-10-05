from pathlib import Path

import pandas as pd

from football_agent.predictor.config import DatasetConfig
from football_agent.predictor.evaluation import temporal_split
from football_agent.predictor.features import FEATURE_COLUMNS, build_features
from football_agent.predictor.prepare import PARTS, load_dataset


def test_prepare_dataset_writes_each_part_and_the_config_that_made_it(
    dataset_dir: Path, dataset_config: DatasetConfig
) -> None:
    assert sorted(path.name for path in dataset_dir.iterdir()) == [
        "dataset.yaml",
        "test.parquet",
        "train.parquet",
        "validation.parquet",
    ]
    config, parts = load_dataset(dataset_dir)
    assert config == dataset_config
    assert set(parts) == set(PARTS)
    assert sorted(parts["train"]["season"].unique()) == [2016, 2017]
    assert list(parts["validation"]["season"].unique()) == [2018]
    assert list(parts["test"]["season"].unique()) == [2019]
    assert set(FEATURE_COLUMNS) <= set(parts["train"].columns)


def test_load_dataset_gives_back_exactly_the_prepared_rows(
    dataset_dir: Path, league: pd.DataFrame, dataset_config: DatasetConfig
) -> None:
    _, parts = load_dataset(dataset_dir)

    expected = temporal_split(build_features(league, window=5), dataset_config.split)
    for part, rows in zip(PARTS, expected, strict=True):
        pd.testing.assert_frame_equal(parts[part], rows.reset_index(drop=True))
