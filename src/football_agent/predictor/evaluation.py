import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from football_agent.predictor.config import Split

# Alphabetical order: scikit-learn metrics read probability columns in sorted label order,
# so every probability array in the project uses columns (away, draw, home).
OUTCOMES = ["A", "D", "H"]
# 95% of a normal distribution lies within 1.96 standard deviations of its mean.
Z_95 = 1.96


def temporal_split(
    matches: pd.DataFrame, split: Split
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matches by whole seasons: train, then validation, then test."""
    train = matches[matches["season"].isin(split.train.to_range())]
    validation = matches[matches["season"].isin(split.validation.to_range())]
    test = matches[matches["season"].isin(split.test.to_range())]
    return train, validation, test


def naive_probabilities(train: pd.DataFrame, n_matches: int) -> np.ndarray:
    """Give each of `n_matches` matches the outcome frequencies observed in `train`."""
    frequencies = train["result"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    return np.tile(frequencies, (n_matches, 1))


def market_probabilities(matches: pd.DataFrame) -> np.ndarray:
    """Turn average closing odds into outcome probabilities, removing the bookmaker margin."""
    odds = matches[["average_odds_away", "average_odds_draw", "average_odds_home"]].to_numpy()
    implied = 1 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def evaluate(probabilities: np.ndarray, results: pd.Series) -> dict[str, float]:
    predicted = np.array(OUTCOMES)[probabilities.argmax(axis=1)]
    return {
        "log_loss": log_loss(results, probabilities, labels=OUTCOMES),
        "accuracy": accuracy_score(results, predicted),
    }


def match_log_losses(probabilities: np.ndarray, results: pd.Series) -> np.ndarray:
    """Log-loss of each match: minus the log of the probability given to its actual result."""
    columns = results.map({outcome: column for column, outcome in enumerate(OUTCOMES)})
    return -np.log(probabilities[np.arange(len(results)), columns.to_numpy()])


def paired_difference(baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Mean of the per-match differences `candidate - baseline` and the half-width of its 95%
    confidence interval. A negative mean means the candidate predicts better."""
    differences = candidate - baseline
    return differences.mean(), Z_95 * differences.std(ddof=1) / math.sqrt(len(differences))


def summarize_folds(scores: pd.DataFrame) -> dict[str, float]:
    """Metrics of a cross-validation from its per-match `scores`: the model's log-loss on each
    validation season, then over every scored match the log-loss of the model and of both
    baselines, and the share of the gap closed."""
    means = scores[["log_loss", "naive_log_loss", "market_log_loss"]].mean().to_dict()
    per_season = scores.groupby("season")["log_loss"].mean()
    return {
        **{f"cv_log_loss_{season}": float(loss) for season, loss in per_season.items()},
        **{f"cv_{name}": float(value) for name, value in means.items()},
        "cv_gap_closed": gap_closed(means),
    }


def compare_with_baselines(
    probabilities: np.ndarray, train: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, float]:
    """Score a model's `probabilities` on `matches`, along with the naive and market baselines.

    The naive baseline uses the outcome frequencies of `train`, the seasons the model learned from.
    """
    results = matches["result"]
    return {
        **evaluate(probabilities, results),
        "naive_log_loss": evaluate(naive_probabilities(train, len(matches)), results)["log_loss"],
        "market_log_loss": evaluate(market_probabilities(matches), results)["log_loss"],
    }


def gap_closed(scores: dict[str, float]) -> float:
    """Share of the log-loss gap between the naive baseline and the market closed by the model."""
    naive = scores["naive_log_loss"]
    return (naive - scores["log_loss"]) / (naive - scores["market_log_loss"])
