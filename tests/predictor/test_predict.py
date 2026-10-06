from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from football_agent.predictor.config import TrainingConfig
from football_agent.predictor.features import FEATURE_COLUMNS, STATS, build_features
from football_agent.predictor.predict import (
    STAT_LABELS,
    MatchPredictor,
    readable_form,
    upcoming_match_features,
)
from football_agent.predictor.prepare import load_dataset
from football_agent.predictor.train import train_model

PSG = {"id": 524, "shortName": "PSG"}
LYON = {"id": 523, "shortName": "Olympique Lyon"}


def test_upcoming_match_features_equal_the_training_features_of_the_same_match(
    league: pd.DataFrame,
) -> None:
    played = league.iloc[100]

    upcoming = upcoming_match_features(
        league, played["home_team"], played["away_team"], played["date"].date(), window=5
    )

    expected = build_features(league, window=5).iloc[[100]]
    pd.testing.assert_frame_equal(
        upcoming[FEATURE_COLUMNS].reset_index(drop=True),
        expected[FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_upcoming_match_features_ignore_matches_played_after_kickoff(
    league: pd.DataFrame,
) -> None:
    kickoff = date(2018, 1, 1)
    later = league[league["date"] > pd.Timestamp(kickoff)].assign(home_goals=9, away_goals=9)
    changed_future = pd.concat([league[league["date"] <= pd.Timestamp(kickoff)], later])

    before = upcoming_match_features(league, "PSG", "Lyon", kickoff, window=5)
    after = upcoming_match_features(changed_future, "PSG", "Lyon", kickoff, window=5)

    pd.testing.assert_frame_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


@pytest.fixture
def predictor(
    dataset_dir: Path, training_config: TrainingConfig, league: pd.DataFrame
) -> MatchPredictor:
    _, parts = load_dataset(dataset_dir)
    model = train_model(parts["train"], parts["validation"], training_config)
    return MatchPredictor(
        model=model,
        form_window=5,
        history=league,
        team_names={524: "PSG", 523: "Lyon", 999: "Bastia"},
    )


def test_predict_gives_outcome_probabilities_and_each_team_recent_form(
    predictor: MatchPredictor,
) -> None:
    prediction = predictor.predict(PSG, LYON, date(2019, 9, 1))

    total = prediction["home_win"] + prediction["draw"] + prediction["away_win"]
    assert total == pytest.approx(1.0, abs=0.002)
    assert prediction["recent_form_matches"] == 5
    assert list(prediction["recent_form"]) == ["PSG", "Olympique Lyon"]
    psg_form = prediction["recent_form"]["PSG"]
    assert set(psg_form) == set(STAT_LABELS.values())
    assert all(isinstance(value, float) for value in psg_form.values())


def test_predict_turns_missing_history_into_null_statistics(predictor: MatchPredictor) -> None:
    bastia = {"id": 999, "shortName": "Bastia"}

    prediction = predictor.predict(bastia, LYON, date(2019, 9, 1))

    assert prediction["recent_form"]["Bastia"]["points_per_match"] is None
    assert prediction["recent_form"]["Olympique Lyon"]["points_per_match"] is not None


def test_every_statistic_has_a_name_for_the_llm() -> None:
    assert set(STAT_LABELS) == set(STATS)


def test_readable_form_groups_features_by_team_under_explicit_names() -> None:
    features = pd.Series(
        {
            "home_points_avg": 1.6,
            "away_points_avg": 0.6,
            "home_shots_on_target_against_avg": 3.4,
            "away_shots_on_target_against_avg": float("nan"),
        }
    )

    form = readable_form(features, "PSG", "Marseille")

    assert form == {
        "PSG": {"points_per_match": 1.6, "shots_on_target_conceded_per_match": 3.4},
        "Marseille": {"points_per_match": 0.6, "shots_on_target_conceded_per_match": None},
    }


def test_predict_refuses_a_team_missing_from_the_team_names(predictor: MatchPredictor) -> None:
    bastia = {"id": 1234, "shortName": "Bastia"}

    with pytest.raises(ValueError, match="No match history for Bastia"):
        predictor.predict(bastia, LYON, date(2019, 9, 1))
