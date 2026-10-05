import numpy as np
import pandas as pd
import pytest

from football_agent.predictor.features import (
    FEATURE_COLUMNS,
    add_rolling_means,
    build_features,
    team_match_rows,
)


def make_matches(games: list[tuple[str, str, str, int, int]]) -> pd.DataFrame:
    """Build a matches table from (date, home team, away team, home goals, away goals).

    Shots on target are set to goals + 3, which keeps hand-computed averages easy.
    """
    matches = pd.DataFrame(
        games, columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    return matches.assign(
        date=pd.to_datetime(matches["date"]),
        home_shots_on_target=matches["home_goals"] + 3.0,
        away_shots_on_target=matches["away_goals"] + 3.0,
    )


PSG_SEASON = make_matches(
    [
        ("2026-08-01", "PSG", "Lyon", 2, 0),
        ("2026-08-08", "Lille", "PSG", 1, 1),
        ("2026-08-15", "PSG", "Nice", 0, 1),
        ("2026-08-22", "Brest", "PSG", 0, 3),
        ("2026-08-29", "PSG", "Lens", 2, 1),
    ]
)


def test_team_match_rows_gives_each_match_from_both_sides() -> None:
    rows = team_match_rows(make_matches([("2026-09-04", "PSG", "Monaco", 1, 2)]))

    psg = rows[rows["team"] == "PSG"].iloc[0]
    monaco = rows[rows["team"] == "Monaco"].iloc[0]
    assert (psg["side"], psg["goals_for"], psg["goals_against"], psg["points"]) == ("home", 1, 2, 0)
    assert (psg["shots_on_target_for"], psg["shots_on_target_against"]) == (4.0, 5.0)
    assert (monaco["side"], monaco["goals_for"], monaco["points"]) == ("away", 2, 3)
    assert (monaco["shots_on_target_for"], monaco["shots_on_target_against"]) == (5.0, 4.0)


def test_add_rolling_means_averages_previous_matches_only() -> None:
    # PSG points: 3, 1, 0, 3, 3. PSG goals scored: 2, 1, 0, 3, 2.
    rows = add_rolling_means(team_match_rows(PSG_SEASON), window=2)

    psg = rows[rows["team"] == "PSG"]
    assert psg["points_avg"].isna().iloc[0]
    assert psg["points_avg"].iloc[1:].tolist() == pytest.approx([3.0, 2.0, 0.5, 1.5])
    assert psg["goals_for_avg"].iloc[1:].tolist() == pytest.approx([2.0, 1.5, 0.5, 1.5])


def test_add_rolling_means_skips_missing_shots() -> None:
    matches = PSG_SEASON.copy()
    matches.loc[1, ["home_shots_on_target", "away_shots_on_target"]] = np.nan

    rows = add_rolling_means(team_match_rows(matches), window=2)

    psg_shots = rows[rows["team"] == "PSG"]["shots_on_target_for_avg"].tolist()
    # Before match 3, PSG's last two matches are 5 shots on target, then a missing value.
    assert psg_shots[2] == 5.0


def test_build_features_puts_each_team_average_on_its_side() -> None:
    matches = make_matches(
        [
            ("2026-08-01", "PSG", "Lyon", 2, 0),
            ("2026-08-08", "Lyon", "PSG", 1, 1),
        ]
    )

    features = build_features(matches, window=5)

    second = features.iloc[1]
    assert (second["home_points_avg"], second["away_points_avg"]) == (0.0, 3.0)
    assert (second["home_goals_against_avg"], second["away_goals_against_avg"]) == (2.0, 0.0)
    assert features.iloc[0][FEATURE_COLUMNS].isna().all()


def test_features_of_a_match_never_depend_on_its_result_or_later_ones() -> None:
    matches = make_matches(
        [
            ("2026-08-01", "PSG", "Lyon", 2, 0),
            ("2026-08-01", "Nice", "Lens", 1, 1),
            ("2026-08-08", "Lyon", "Nice", 0, 1),
            ("2026-08-08", "Lens", "PSG", 2, 2),
            ("2026-08-15", "PSG", "Nice", 3, 1),
            ("2026-08-15", "Lyon", "Lens", 1, 0),
            ("2026-08-22", "Nice", "PSG", 0, 0),
            ("2026-08-22", "Lens", "Lyon", 2, 1),
            ("2026-08-29", "Lyon", "PSG", 1, 2),
            ("2026-08-29", "Lens", "Nice", 0, 3),
        ]
    )
    changed_match = 4
    changed = matches.copy()
    changed.loc[
        changed_match, ["home_goals", "away_goals", "home_shots_on_target", "away_shots_on_target"]
    ] = [0, 5, 1.0, 9.0]

    before = build_features(matches, window=5)[FEATURE_COLUMNS]
    after = build_features(changed, window=5)[FEATURE_COLUMNS]

    earlier_or_same_day = matches["date"] <= matches.loc[changed_match, "date"]
    assert before[earlier_or_same_day].equals(after[earlier_or_same_day])
    for column in FEATURE_COLUMNS:
        assert not before.loc[~earlier_or_same_day, column].equals(
            after.loc[~earlier_or_same_day, column]
        ), column
