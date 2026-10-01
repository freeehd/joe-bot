import unittest

import numpy as np

from models.train_v2 import confidence_bucket_report, evaluate_probabilities, expected_calibration_error


class TrainV2MetricTests(unittest.TestCase):
    def test_metrics_are_finite_for_three_class_probabilities(self):
        y = np.array([0, 1, 2, 1, 2, 0])
        p = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
        ])
        metrics = evaluate_probabilities(y, p)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["long_pr_auc"], 0.9)
        self.assertGreater(metrics["short_pr_auc"], 0.9)
        self.assertGreaterEqual(metrics["ece_10"], 0.0)
        self.assertLessEqual(metrics["ece_10"], 1.0)
        self.assertGreater(len(metrics["confidence_buckets"]), 0)

    def test_ece_is_zero_for_perfectly_confident_correct_model(self):
        y = np.array([0, 1, 2])
        p = np.eye(3)
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0)

    def test_confidence_bucket_counts_all_rows(self):
        y = np.array([0, 1, 2, 0])
        p = np.array([
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
        ])
        rows = confidence_bucket_report(y, p)
        self.assertEqual(sum(row["samples"] for row in rows), 4)


if __name__ == "__main__":
    unittest.main()
