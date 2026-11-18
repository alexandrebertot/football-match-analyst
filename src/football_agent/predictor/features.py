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
AVERAGE_COLUMNS = [f"{side}_{stat}_avg" for stat in STATS for side in SIDES]
ELO_COLUMNS = [f"{side}_elo" for side in SIDES]
FEATURE_COLUMNS = [*AVERAGE_COLUMNS, *ELO_COLUMNS]
ELO_INITIAL = 1500
# K and H minimise the Elo's own log-loss over 2010-11 to 2018-19: every fold of the
# cross-validation trains on these seasons and none scores them.
ELO_K = 15
ELO_HOME_ADVANTAGE = 60


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


def expected_home_score(home_rating: float, away_rating: float) -> float:
    """Score the home team is expected to get (win 1, draw 0.5, loss 0) from both Elo ratings."""
    return 1 / (1 + 10 ** ((away_rating - home_rating - ELO_HOME_ADVANTAGE) / 400))


def elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Both teams' Elo ratings before each match, computed league by league, season by season.

    Every team of a league's first season starts at ELO_INITIAL, and a promoted team starts its
    season at the mean rating of the teams that left the league. A match without a score changes
    no rating, and a match without a season (the one to predict) gets the teams' latest ratings.
    """
    before: dict[int, tuple[float, float]] = {}
    latest: dict[str, float] = {}
    for _, league in matches.groupby("competition"):
        ratings: dict[str, float] = {}
        previous_teams: set[str] = set()
        for _, season_matches in league.groupby("season"):
            games = season_matches.sort_values("date", kind="stable")
            teams = set(games["home_team"]) | set(games["away_team"])
            departed = previous_teams - teams
            start = np.mean([ratings[team] for team in departed]) if departed else ELO_INITIAL
            for team in teams - previous_teams:
                ratings[team] = start
            columns = ["home_team", "away_team", "home_goals", "away_goals"]
            for match, home, away, home_goals, away_goals in games[columns].itertuples():
                before[match] = (ratings[home], ratings[away])
                if pd.isna(home_goals) or pd.isna(away_goals):
                    continue
                # 1 for a home win, 0.5 for a draw, 0 for an away win.
                score = (np.sign(home_goals - away_goals) + 1) / 2
                change = ELO_K * (score - expected_home_score(ratings[home], ratings[away]))
                ratings[home] += change
                ratings[away] -= change
            previous_teams = teams
        latest.update(ratings)
    unseasoned = matches[matches["season"].isna() | matches["competition"].isna()]
    for match, home, away in unseasoned[["home_team", "away_team"]].itertuples():
        before[match] = (latest.get(home, np.nan), latest.get(away, np.nan))
    ratings_before = pd.DataFrame.from_dict(before, orient="index", columns=ELO_COLUMNS)
    return ratings_before.reindex(matches.index)


def build_features(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add both teams' recent averages and Elo ratings to each match, using only matches played
    before it."""
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
    ).join(elo_ratings(matches))
