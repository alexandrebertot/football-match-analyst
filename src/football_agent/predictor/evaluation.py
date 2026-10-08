import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from football_agent.predictor.config import Split

# Alphabetical order: scikit-learn metrics read probability columns in sorted label order,
# so every probability array in the project uses columns (away, draw, home).
OUTCOMES = ["A", "D", "H"]


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
