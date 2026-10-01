import unittest

import pandas as pd

from labels.triple_barrier import BarrierConfig, create_multiclass_labels
from research.barrier_sweep import summarize_labels


class BarrierSweepTests(unittest.TestCase):
    def test_summary_rates_sum_to_one(self):
        index = pd.date_range("2026-01-05 14:30Z", periods=30, freq="min")
        close = pd.Series([100 + ((i % 6) - 3) * 0.12 for i in range(30)], index=index)
        raw = pd.DataFrame({
            "open": close,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": 1000,
            "vwap": close,
        }, index=index)
        labeled = create_multiclass_labels(
            raw,
            BarrierConfig(horizon_bars=5, target_pct=0.002, stop_pct=0.001),
        )
        summary = summarize_labels(labeled)
        self.assertGreater(summary["valid_rows"], 0)
        total_rate = summary["wait_rate"] + summary["long_rate"] + summary["short_rate"]
        self.assertAlmostEqual(total_rate, 1.0)
        self.assertGreaterEqual(summary["directional_ambiguous_rate"], 0.0)
        self.assertLessEqual(summary["directional_ambiguous_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
