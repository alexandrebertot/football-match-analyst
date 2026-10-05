from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from football_agent.predictor.data import load_history
from football_agent.predictor.evaluation import OUTCOMES
from football_agent.predictor.features import build_features
from football_agent.predictor.registry import load_champion
from football_agent.predictor.teams import load_team_names


def upcoming_match_features(
    history: pd.DataFrame, home_team: str, away_team: str, kickoff: date, window: int
) -> pd.DataFrame:
    """Features of a match not played yet (one row), from the matches played before `kickoff`.

    The match is added to the history without a score and goes through the same feature code
    as training, so training and prediction can never compute features differently.
    """
    played = history[history["date"] < pd.Timestamp(kickoff)]
    upcoming = pd.DataFrame(
        [{"date": pd.Timestamp(kickoff), "home_team": home_team, "away_team": away_team}]
    )
    matches = pd.concat([played, upcoming], ignore_index=True)
    # iloc[[-1]] keeps a one-row DataFrame, and so the numeric type of every feature column.
    return build_features(matches, window).iloc[[-1]]


@dataclass
class MatchPredictor:
    model: LGBMClassifier
    form_window: int
    history: pd.DataFrame
    team_names: dict[int, str]

    def csv_team_name(self, team_id: int, team: str) -> str:
        if team_id not in self.team_names:
            raise ValueError(f"No match history for {team}: predictions are not available for it.")
        return self.team_names[team_id]

    def predict(self, home: dict[str, Any], away: dict[str, Any], kickoff: date) -> dict[str, Any]:
        """Predict a match between two football-data.org teams (with "id" and "shortName")."""
        features = upcoming_match_features(
            self.history,
            self.csv_team_name(home["id"], home["shortName"]),
            self.csv_team_name(away["id"], away["shortName"]),
            kickoff,
            self.form_window,
        )
        model_features = features[self.model.feature_name_]
        probabilities = dict(
            zip(OUTCOMES, self.model.predict_proba(model_features)[0], strict=True)
        )
        return {
            "home_win": round(float(probabilities["H"]), 3),
            "draw": round(float(probabilities["D"]), 3),
            "away_win": round(float(probabilities["A"]), 3),
            # NaN is not valid JSON for the LLM: a team without history gets null instead.
            "features": {
                name: None if pd.isna(value) else round(float(value), 2)
                for name, value in model_features.iloc[0].items()
            },
        }


def load_predictor(data_dir: Path, today: date) -> MatchPredictor:
    """Load the champion model, its form window, the match history and the team names."""
    model, dataset_config = load_champion()
    return MatchPredictor(
        model=model,
        form_window=dataset_config.features.form_window,
        history=load_history(data_dir, today),
        team_names=load_team_names(),
    )
