import numpy as np
import pandas as pd
import pytest

from football_agent.predictor.features import (
    FEATURE_COLUMNS,
    add_rolling_form,
    build_features,
    team_match_rows,
)


def make_matches(games: list[tuple[str, str, str, int, int]]) -> pd.DataFrame:
    """Build a matches table from (date, home team, away team, home goals, away goals)."""
    return pd.DataFrame(
        games, columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
    ).assign(date=lambda matches: pd.to_datetime(matches["date"]))


def test_team_match_rows_gives_each_match_from_both_sides() -> None:
    rows = team_match_rows(make_matches([("2026-09-04", "PSG", "Monaco", 1, 2)]))

    psg = rows[rows["team"] == "PSG"].iloc[0]
    monaco = rows[rows["team"] == "Monaco"].iloc[0]
    assert (psg["side"], psg["goals_for"], psg["goals_against"], psg["points"]) == ("home", 1, 2, 0)
    assert (monaco["side"], monaco["goals_for"], monaco["points"]) == ("away", 2, 3)


def test_add_rolling_form_averages_previous_matches_only() -> None:
    # PSG: win, draw, loss, win, win -> points 3, 1, 0, 3, 3.
    rows = team_match_rows(
        make_matches(
            [
                ("2026-08-01", "PSG", "Lyon", 2, 0),
                ("2026-08-08", "Lille", "PSG", 1, 1),
                ("2026-08-15", "PSG", "Nice", 0, 1),
                ("2026-08-22", "Brest", "PSG", 0, 3),
                ("2026-08-29", "PSG", "Lens", 2, 1),
            ]
        )
    )

    form = add_rolling_form(rows, window=2)

    psg_form = form[form["team"] == "PSG"]["form_points"].tolist()
    assert np.isnan(psg_form[0])
    assert psg_form[1:] == pytest.approx([3.0, 2.0, 0.5, 1.5])


def test_build_features_puts_each_team_form_on_its_side() -> None:
    matches = make_matches(
        [
            ("2026-08-01", "PSG", "Lyon", 2, 0),
            ("2026-08-08", "Lyon", "PSG", 1, 1),
        ]
    )

    features = build_features(matches)

    second = features.iloc[1]
    assert (second["home_form_points"], second["away_form_points"]) == (0.0, 3.0)
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
    changed.loc[changed_match, ["home_goals", "away_goals"]] = [0, 5]

    before = build_features(matches)[FEATURE_COLUMNS]
    after = build_features(changed)[FEATURE_COLUMNS]

    earlier_or_same_day = matches["date"] <= matches.loc[changed_match, "date"]
    assert before[earlier_or_same_day].equals(after[earlier_or_same_day])
    assert not before[~earlier_or_same_day].equals(after[~earlier_or_same_day])
