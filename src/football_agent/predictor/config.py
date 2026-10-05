from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from football_agent.predictor.features import FEATURE_COLUMNS


class StrictModel(BaseModel):
    # A typo in a YAML file must fail instead of silently falling back to a default.
    model_config = ConfigDict(extra="forbid")


class SeasonRange(StrictModel):
    """Seasons by start year, both ends included: 2016 to 2022 covers 2016-17 to 2022-23."""

    first: int
    last: int

    @model_validator(mode="after")
    def check_order(self) -> "SeasonRange":
        if self.last < self.first:
            raise ValueError(f"last season {self.last} is before first season {self.first}")
        return self

    def to_range(self) -> range:
        return range(self.first, self.last + 1)


class Split(StrictModel):
    train: SeasonRange
    validation: SeasonRange
    test: SeasonRange

    @model_validator(mode="after")
    def check_chronological(self) -> "Split":
        if not self.train.last < self.validation.first <= self.validation.last < self.test.first:
            raise ValueError("train, validation and test seasons must follow each other in time")
        return self


class FeatureOptions(StrictModel):
    form_window: int = Field(gt=0)


class DatasetConfig(StrictModel):
    name: str
    seasons: SeasonRange
    split: Split
    features: FeatureOptions

    @model_validator(mode="after")
    def check_split_within_seasons(self) -> "DatasetConfig":
        if self.split.train.first < self.seasons.first or self.split.test.last > self.seasons.last:
            raise ValueError("the split must stay within the loaded seasons")
        return self


class RunOptions(StrictModel):
    name: str


class ModelParams(StrictModel):
    """LightGBM hyperparameters, all required so that every config states them explicitly."""

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(gt=1)
    max_depth: int
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    subsample_freq: int = Field(ge=0)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)


class TrainingOptions(StrictModel):
    early_stopping_rounds: int | None = Field(gt=0)


class TrainingConfig(StrictModel):
    run: RunOptions
    dataset: str
    features: list[str] = Field(min_length=1)
    model: ModelParams
    training: TrainingOptions

    @field_validator("features")
    @classmethod
    def check_features_exist(cls, features: list[str]) -> list[str]:
        unknown = set(features) - set(FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown features {sorted(unknown)}; available: {FEATURE_COLUMNS}")
        return features


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dataset_config(path: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(read_yaml(path))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(read_yaml(path))
