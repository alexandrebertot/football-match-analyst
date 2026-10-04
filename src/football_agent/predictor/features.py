import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["home_form_points", "away_form_points"]


def team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team and match: each match appears twice, once from each side."""
    sides = []
    for side, team, goals_for, goals_against in [
        ("home", "home_team", "home_goals", "away_goals"),
        ("away", "away_team", "away_goals", "home_goals"),
    ]:
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches.index,
                    "date": matches["date"],
                    "side": side,
                    "team": matches[team],
                    "goals_for": matches[goals_for],
                    "goals_against": matches[goals_against],
                }
            )
        )
    rows = pd.concat(sides, ignore_index=True)
    rows["points"] = np.select(
        [rows["goals_for"] > rows["goals_against"], rows["goals_for"] == rows["goals_against"]],
        [3, 1],
        0,
    )
    return rows.sort_values(["team", "date"], kind="stable", ignore_index=True)


def add_rolling_form(rows: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add each team's average points over its previous `window` matches (current one excluded)."""
    # shift(1) is what prevents leakage: a match only sees the results of the matches before it.
    form = rows.groupby("team")["points"].transform(
        lambda points: points.shift(1).rolling(window, min_periods=1).mean()
    )
    return rows.assign(form_points=form)


def build_features(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add the recent form of both teams to each match, using only matches played before it."""
    rows = add_rolling_form(team_match_rows(matches), window)
    form = rows.pivot(index="match_id", columns="side", values="form_points")
    return matches.assign(home_form_points=form["home"], away_form_points=form["away"])
