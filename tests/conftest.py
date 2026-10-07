import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from football_agent.data_api import open_cache


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = open_cache(tmp_path / "cache" / "responses.sqlite")
    yield db
    db.close()


class FakePredictor:
    """Stand-in for MatchPredictor: returns a fixed prediction and records each request."""

    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, Any], dict[str, Any], date]] = []

    def predict(self, home: dict[str, Any], away: dict[str, Any], kickoff: date) -> dict[str, Any]:
        self.requests.append((home, away, kickoff))
        return {
            "home_win": 0.5,
            "draw": 0.3,
            "away_win": 0.2,
            "recent_form_matches": 5,
            "recent_form": {
                home["shortName"]: {"points_per_match": 2.2, "shots_on_target_per_match": 5.4},
                away["shortName"]: {"points_per_match": 1.4, "shots_on_target_per_match": 3.8},
            },
        }


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()
