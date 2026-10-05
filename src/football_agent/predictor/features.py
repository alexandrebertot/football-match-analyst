import numpy as np
import pandas as pd

# Per-match statistics of a team, averaged over its previous matches to build the features.
STATS = [
    "points",
    "goals_for",
    "goals_against",
    "shots_on_target_for",
    "shots_on_target_against",
]
SIDES = ["home", "away"]
FEATURE_COLUMNS = [f"{side}_{stat}_avg" for stat in STATS for side in SIDES]


def team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team and match: each match appears twice, once from each side."""
    sides = []
    for side, opponent in [("home", "away"), ("away", "home")]:
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches.index,
                    "date": matches["date"],
                    "side": side,
                    "team": matches[f"{side}_team"],
                    "goals_for": matches[f"{side}_goals"],
                    "goals_against": matches[f"{opponent}_goals"],
                    "shots_on_target_for": matches[f"{side}_shots_on_target"],
                    "shots_on_target_against": matches[f"{opponent}_shots_on_target"],
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


def add_rolling_means(rows: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add each team's average of every stat over its previous `window` matches (current excluded).

    Missing values (2 awarded matches have no shots) are skipped by the average.
    """
    # shift(1) is what prevents leakage: a match only sees the matches played before it.
    means = rows.groupby("team")[STATS].transform(
        lambda stats: stats.shift(1).rolling(window, min_periods=1).mean()
    )
    return rows.assign(**{f"{stat}_avg": means[stat] for stat in STATS})


def build_features(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add both teams' recent averages to each match, using only matches played before it."""
    rows = add_rolling_means(team_match_rows(matches), window)
    averages = rows.pivot(
        index="match_id", columns="side", values=[f"{stat}_avg" for stat in STATS]
    )
    return matches.assign(
        **{
            f"{side}_{stat}_avg": averages[(f"{stat}_avg", side)]
            for stat in STATS
            for side in SIDES
        }
    )
