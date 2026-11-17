import argparse
import io
from typing import Any

import mlflow
import pandas as pd

from football_agent.predictor.evaluation import paired_difference
from football_agent.predictor.registry import latest_run
from football_agent.predictor.train import MATCH_COLUMNS, MATCH_SCORES_FILE


def compare_scores(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Compare two cross-validations match by match.

    Returns the mean log-loss difference `candidate - baseline` (negative when the candidate is
    better), the half-width of its 95% confidence interval, the mean difference on each season,
    and whether the candidate is kept: only when the whole interval is below zero.
    """
    if not baseline[MATCH_COLUMNS].equals(candidate[MATCH_COLUMNS]):
        raise ValueError("The two runs did not score the same matches: use the same dataset.")
    difference, half_width = paired_difference(
        baseline["log_loss"].to_numpy(), candidate["log_loss"].to_numpy()
    )
    per_season = (candidate["log_loss"] - baseline["log_loss"]).groupby(baseline["season"]).mean()
    return {
        "difference": difference,
        "half_width": half_width,
        "per_season": per_season.to_dict(),
        "keep_candidate": difference + half_width < 0,
    }


def match_scores(run_name: str) -> tuple[str, pd.DataFrame]:
    """Return the id of the latest run named `run_name` and the per-match scores it recorded."""
    run_id = latest_run(run_name).info.run_id
    scores_csv = mlflow.artifacts.load_text(f"runs:/{run_id}/{MATCH_SCORES_FILE}")
    return run_id, pd.read_csv(io.StringIO(scores_csv))


def compare_runs(baseline_name: str, candidate_name: str) -> dict[str, Any]:
    """Compare the latest runs with these names, and record the comparison in the candidate run."""
    baseline_id, baseline_scores = match_scores(baseline_name)
    candidate_id, candidate_scores = match_scores(candidate_name)
    comparison = compare_scores(baseline_scores, candidate_scores)
    with mlflow.start_run(run_id=candidate_id):
        mlflow.log_metrics(
            {
                "paired_difference": comparison["difference"],
                "paired_half_width": comparison["half_width"],
            }
        )
        mlflow.set_tags(
            {
                "compared_with": f"{baseline_name} ({baseline_id})",
                "decision": "keep" if comparison["keep_candidate"] else "reject",
            }
        )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two training runs match by match.")
    parser.add_argument("--baseline", required=True, help="name of the reference training run")
    parser.add_argument("--candidate", required=True, help="name of the training run to judge")
    arguments = parser.parse_args()
    comparison = compare_runs(arguments.baseline, arguments.candidate)
    print(
        f"Log-loss difference: {comparison['difference']:+.4f} ± {comparison['half_width']:.4f} "
        "(95% confidence interval, negative is better)"
    )
    for season, difference in comparison["per_season"].items():
        print(f"  {season}-{(season + 1) % 100:02d}: {difference:+.4f}")
    print("Keep the candidate." if comparison["keep_candidate"] else "Keep the baseline.")
