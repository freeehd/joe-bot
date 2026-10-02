import unittest

import pandas as pd

from labels.triple_barrier import BarrierConfig
from research.label_tournament import _selection_cutoff, evaluate_pretest_config


class FakeLake:
    def __init__(self, raw):
        self.raw = raw
    def load_raw_bars(self, version, symbol):
        return self.raw.copy()


class LabelTournamentTests(unittest.TestCase):
    def test_selection_cutoff_is_calibration_end(self):
        cutoff = _selection_cutoff({"chronological_periods": {"calibration": {"end_inclusive": "2026-01-05T15:09:00+00:00"}}})
        self.assertEqual(cutoff, pd.Timestamp("2026-01-05T15:09:00Z"))

    def test_evaluation_truncates_future_test_rows_before_labeling(self):
        idx = pd.date_range("2026-01-05 14:30Z", periods=60, freq="min")
        close = pd.Series([100 + i * 0.03 for i in range(len(idx))], index=idx)
        raw = pd.DataFrame({
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "volume": 1000, "vwap": close,
        }, index=idx)
        cutoff = idx[39]
        result = evaluate_pretest_config(
            FakeLake(raw), raw_snapshot_version="v1", symbols=["AAA"],
            config=BarrierConfig(horizon_bars=5), cutoff=cutoff,
        )
        self.assertEqual(result["selection_window_end"], cutoff.isoformat())
        self.assertLessEqual(result["valid_rows"], 35)


if __name__ == "__main__":
    unittest.main()
