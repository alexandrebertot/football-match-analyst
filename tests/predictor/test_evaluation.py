import math

import numpy as np
import pandas as pd
import pytest

from football_agent.predictor.evaluation import (
    evaluate,
    market_probabilities,
    naive_probabilities,
    temporal_split,
)


def test_temporal_split_keeps_seasons_in_order_without_overlap() -> None:
    matches = pd.DataFrame({"season": range(2016, 2026)})

    train, validation, test = temporal_split(matches)

    assert list(train["season"]) == list(range(2016, 2023))
    assert list(validation["season"]) == [2023]
    assert list(test["season"]) == [2024, 2025]


def test_naive_probabilities_repeat_train_frequencies_in_away_draw_home_order() -> None:
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})

    probabilities = naive_probabilities(train, n_matches=3)

    assert probabilities.shape == (3, 3)
    assert probabilities[0] == pytest.approx([0.25, 0.25, 0.5])
    assert (probabilities == probabilities[0]).all()


def test_market_probabilities_remove_the_bookmaker_margin() -> None:
    # Liverpool - Bournemouth, 2025-08-15: average closing odds 1.29 / 6.02 / 8.68.
    matches = pd.DataFrame(
        {"average_odds_home": [1.29], "average_odds_draw": [6.02], "average_odds_away": [8.68]}
    )

    probabilities = market_probabilities(matches)

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx([0.109, 0.157, 0.734], abs=0.001)


def test_evaluate_matches_a_hand_computed_log_loss() -> None:
    # Columns are (away, draw, home): a home win given 0.7, then an away win given 0.2.
    probabilities = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    results = pd.Series(["H", "A"])

    scores = evaluate(probabilities, results)

    assert scores["log_loss"] == pytest.approx(-(math.log(0.7) + math.log(0.2)) / 2)
    assert scores["accuracy"] == 0.5


def test_evaluate_gives_ln3_for_uniform_predictions() -> None:
    results = pd.Series(["H", "D", "A", "H"])

    scores = evaluate(np.full((4, 3), 1 / 3), results)

    assert scores["log_loss"] == pytest.approx(math.log(3))
