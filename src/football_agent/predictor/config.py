from pathlib import Path
from typing import Any

import yaml
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field, field_validator

from football_agent.predictor.features import FEATURE_COLUMNS


class TrainingConfig(BaseModel):
    # A typo in the YAML must fail instead of silently falling back to a default.
    model_config = ConfigDict(extra="forbid")

    run_name: str
    features: list[str] = Field(min_length=1)
    form_window: int = Field(gt=0)
    model: dict[str, Any] = {}
    early_stopping_rounds: int | None = Field(default=None, gt=0)

    @field_validator("features")
    @classmethod
    def check_features_exist(cls, features: list[str]) -> list[str]:
        unknown = set(features) - set(FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown features {sorted(unknown)}; available: {FEATURE_COLUMNS}")
        return features

    @field_validator("model")
    @classmethod
    def check_model_params_exist(cls, params: dict[str, Any]) -> dict[str, Any]:
        # LightGBM only warns about unknown parameters and then ignores them.
        unknown = set(params) - set(LGBMClassifier().get_params())
        if unknown:
            raise ValueError(f"Unknown LightGBM parameters {sorted(unknown)}")
        return params


def load_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
