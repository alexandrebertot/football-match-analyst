import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# Alphabetical order: scikit-learn metrics and LightGBM both sort class labels this way,
# so every probability array in the project uses columns (away, draw, home).
OUTCOMES = ["A", "D", "H"]
TRAIN_SEASONS = range(2016, 2023)
VALIDATION_SEASONS = range(2023, 2024)
TEST_SEASONS = range(2024, 2026)


def temporal_split(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matches by whole seasons: train, then validation, then test."""
    train = matches[matches["season"].isin(TRAIN_SEASONS)]
    validation = matches[matches["season"].isin(VALIDATION_SEASONS)]
    test = matches[matches["season"].isin(TEST_SEASONS)]
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
