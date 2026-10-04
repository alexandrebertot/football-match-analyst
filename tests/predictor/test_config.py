from pathlib import Path

import pytest
from pydantic import ValidationError

from football_agent.predictor.config import load_config

VALID_CONFIG = """
run_name: baseline-form
features:
  - home_form_points
  - away_form_points
form_window: 5
model:
  learning_rate: 0.05
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_reads_a_valid_file(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.run_name == "baseline-form"
    assert config.features == ["home_form_points", "away_form_points"]
    assert config.form_window == 5
    assert config.model == {"learning_rate": 0.05}


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("form_window: 5", "form_windw: 5"),
        ("form_window: 5", "form_window: -3"),
        ("  - away_form_points", "  - away_form_pts"),
        ("learning_rate: 0.05", "learnig_rate: 0.05"),
    ],
    ids=["unknown-key", "non-positive-window", "unknown-feature", "unknown-model-param"],
)
def test_load_config_rejects_invalid_files(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, VALID_CONFIG.replace(old, new)))


def test_the_baseline_config_of_the_repository_is_valid() -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "baseline.yaml")

    assert config.run_name == "baseline-form"
