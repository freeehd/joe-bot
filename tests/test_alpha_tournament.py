import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.v2 import FEATURE_COLUMNS_V2
from research.alpha_tournament import (
    ABLATION_GROUPS,
    outer_split,
    run_feature_ablation,
    run_model_tournament,
    selection_score,
)


def synthetic_dataset(rows=360):
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    labels = np.resize(np.array([0, 1, 2]), rows)
    data = {name: np.linspace(0.0, 1.0, rows) + (i * 1e-4) for i, name in enumerate(FEATURE_COLUMNS_V2)}
    data["trade_label"] = labels
    data["training_symbol"] = "AAA"
    return pd.DataFrame(data, index=timestamps)


def manifest():
    return {
        "dataset_version": "fixture",
        "feature_engine": "v2",
        "features": list(FEATURE_COLUMNS_V2),
        "label_parameters": {"horizon_bars": 2, "target_pct": 0.003, "stop_pct": 0.0015, "use_atr": False, "atr_period": 14, "atr_target_multiplier": 1.0, "atr_stop_multiplier": 1.0},
        "split_policy": {"train_fraction": 0.70, "calibration_fraction": 0.15},
    }


def factories():
    return {"tiny": lambda: LogisticRegression(max_iter=100, random_state=42)}


class AlphaTournamentTests(unittest.TestCase):
    def test_outer_final_test_is_separate(self):
        train, calibration, final_test = outer_split(synthetic_dataset(), manifest())
        self.assertLess(train.index.max(), calibration.index.min())
        self.assertLess(calibration.index.max(), final_test.index.min())

    def test_selection_score_rewards_better_directional_metrics(self):
        better = {"macro_f1": .6, "long_pr_auc": .7, "short_pr_auc": .7, "log_loss": .8, "ece_10": .05}
        worse = {"macro_f1": .4, "long_pr_auc": .5, "short_pr_auc": .5, "log_loss": 1.2, "ece_10": .15}
        self.assertGreater(selection_score(better), selection_score(worse))

    @patch("research.alpha_tournament.available_candidates", side_effect=factories)
    def test_tournament_marks_final_test_unused_for_ranking(self, _):
        result = run_model_tournament(
            synthetic_dataset(), manifest(), list(FEATURE_COLUMNS_V2),
            model_names=["tiny"], calibration_methods=["sigmoid"],
        )
        self.assertFalse(result["final_test_partition"]["used_for_ranking"])
        self.assertFalse(result["results"][0]["final_test_used"])

    @patch("research.alpha_tournament.available_candidates", side_effect=factories)
    def test_ablation_runs_full_plus_every_declared_group(self, _):
        result = run_feature_ablation(
            synthetic_dataset(), manifest(), list(FEATURE_COLUMNS_V2), model_name="tiny"
        )
        self.assertEqual(len(result), 1 + len(ABLATION_GROUPS))
        names = {row["experiment"] for row in result}
        self.assertIn("full", names)
        self.assertIn("without_breadth", names)


if __name__ == "__main__":
    unittest.main()
