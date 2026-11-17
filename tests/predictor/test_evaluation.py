import math

import numpy as np
import pandas as pd
import pytest

from football_agent.predictor.config import Split
from football_agent.predictor.evaluation import (
    compare_with_baselines,
    evaluate,
    gap_closed,
    market_probabilities,
    match_log_losses,
    naive_probabilities,
    paired_difference,
    temporal_split,
)


def test_temporal_split_keeps_seasons_in_order_without_overlap() -> None:
    matches = pd.DataFrame({"season": range(2016, 2026)})
    split = Split.model_validate(
        {
            "train": {"first": 2016, "last": 2022},
            "validation": {"first": 2023, "last": 2023},
            "test": {"first": 2024, "last": 2025},
        }
    )

    train, validation, test = temporal_split(matches, split)

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


def test_match_log_losses_take_the_actual_result_probability_and_average_to_the_log_loss() -> None:
    probabilities = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    results = pd.Series(["H", "A"])

    losses = match_log_losses(probabilities, results)

    assert losses == pytest.approx([-math.log(0.7), -math.log(0.2)])
    assert losses.mean() == pytest.approx(evaluate(probabilities, results)["log_loss"])


def test_paired_difference_matches_a_hand_computed_interval() -> None:
    # Differences -0.1, 0, -0.2, -0.1: mean -0.1, standard deviation sqrt(0.02 / 3).
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([0.9, 1.0, 0.8, 0.9])

    mean, half_width = paired_difference(baseline, candidate)

    assert mean == pytest.approx(-0.1)
    assert half_width == pytest.approx(1.96 * math.sqrt(0.02 / 3) / 2)


def test_paired_difference_ignores_how_hard_each_match_is() -> None:
    # Losses that vary a lot from match to match, and a candidate better by 0.01 on every one.
    baseline = np.array([0.2, 1.5, 0.7, 2.3, 1.1])
    candidate = baseline - 0.01

    mean, half_width = paired_difference(baseline, candidate)

    assert mean == pytest.approx(-0.01)
    assert half_width == pytest.approx(0.0, abs=1e-12)


def test_compare_with_baselines_scores_the_model_and_both_baselines_on_the_same_matches() -> None:
    # Train frequencies: away 0.25, draw 0.25, home 0.5. Equal odds give the market 1/3 each.
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})
    matches = pd.DataFrame(
        {
            "result": ["H", "A"],
            "average_odds_home": [3.0, 3.0],
            "average_odds_draw": [3.0, 3.0],
            "average_odds_away": [3.0, 3.0],
        }
    )
    probabilities = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])

    scores = compare_with_baselines(probabilities, train, matches)

    assert scores["log_loss"] == pytest.approx(-(math.log(0.7) + math.log(0.2)) / 2)
    assert scores["accuracy"] == 0.5
    assert scores["naive_log_loss"] == pytest.approx(-(math.log(0.5) + math.log(0.25)) / 2)
    assert scores["market_log_loss"] == pytest.approx(math.log(3))


@pytest.mark.parametrize(("log_loss", "expected"), [(1.0185, 0.4746), (1.0774, 0.0), (0.9533, 1.0)])
def test_gap_closed_measures_the_way_from_the_naive_baseline_to_the_market(
    log_loss: float, expected: float
) -> None:
    scores = {"log_loss": log_loss, "naive_log_loss": 1.0774, "market_log_loss": 0.9533}

    assert gap_closed(scores) == pytest.approx(expected, abs=0.0001)
