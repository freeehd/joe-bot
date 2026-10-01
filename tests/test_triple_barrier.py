import unittest

import pandas as pd

from labels.triple_barrier import (
    BarrierConfig,
    BarrierOutcome,
    TradeLabel,
    calculate_atr,
    create_multiclass_labels,
    evaluate_barriers,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "vwap"])


class TripleBarrierTests(unittest.TestCase):
    def test_long_target_before_stop(self):
        df = frame([
            [100, 100.05, 99.95, 100, 1000, 100],
            [100, 100.10, 99.95, 100.05, 1000, 100],
            [100, 100.35, 100.00, 100.30, 1000, 100],
            [100, 100.10, 99.70, 99.80, 1000, 100],
        ])
        outcome, bars = evaluate_barriers(
            df, 0, side="LONG", horizon_bars=3,
            target_distance=0.30, stop_distance=0.15,
        )
        self.assertEqual(outcome, BarrierOutcome.WIN)
        self.assertEqual(bars, 2)

    def test_long_stop_before_target(self):
        df = frame([
            [100, 100.05, 99.95, 100, 1000, 100],
            [100, 100.05, 99.80, 99.90, 1000, 100],
            [100, 100.40, 99.90, 100.30, 1000, 100],
        ])
        outcome, bars = evaluate_barriers(
            df, 0, side="LONG", horizon_bars=2,
            target_distance=0.30, stop_distance=0.15,
        )
        self.assertEqual(outcome, BarrierOutcome.LOSS)
        self.assertEqual(bars, 1)

    def test_same_bar_target_and_stop_is_ambiguous(self):
        df = frame([
            [100, 100.05, 99.95, 100, 1000, 100],
            [100, 100.40, 99.70, 100.05, 1000, 100],
        ])
        outcome, bars = evaluate_barriers(
            df, 0, side="LONG", horizon_bars=1,
            target_distance=0.30, stop_distance=0.15,
        )
        self.assertEqual(outcome, BarrierOutcome.AMBIGUOUS)
        self.assertEqual(bars, 1)

    def test_short_target_before_stop(self):
        df = frame([
            [100, 100.05, 99.95, 100, 1000, 100],
            [100, 100.05, 99.60, 99.70, 1000, 100],
            [100, 100.40, 99.80, 100.30, 1000, 100],
        ])
        outcome, bars = evaluate_barriers(
            df, 0, side="SHORT", horizon_bars=2,
            target_distance=0.30, stop_distance=0.15,
        )
        self.assertEqual(outcome, BarrierOutcome.WIN)
        self.assertEqual(bars, 1)

    def test_no_resolution(self):
        df = frame([
            [100, 100.02, 99.98, 100, 1000, 100],
            [100, 100.10, 99.95, 100.02, 1000, 100],
            [100, 100.08, 99.92, 100.00, 1000, 100],
        ])
        outcome, bars = evaluate_barriers(
            df, 0, side="LONG", horizon_bars=2,
            target_distance=0.30, stop_distance=0.15,
        )
        self.assertEqual(outcome, BarrierOutcome.NO_RESOLUTION)
        self.assertEqual(bars, 2)

    def test_incomplete_horizon_is_not_wait(self):
        df = frame([
            [100, 100.02, 99.98, 100, 1000, 100],
            [100, 100.10, 99.95, 100.02, 1000, 100],
        ])
        labeled = create_multiclass_labels(df, BarrierConfig(horizon_bars=2))
        self.assertFalse(bool(labeled.iloc[0]["label_valid"]))
        self.assertTrue(pd.isna(labeled.iloc[0]["trade_label"]))

    def test_multiclass_long(self):
        df = frame([
            [100, 100.02, 99.98, 100, 1000, 100],
            [100, 100.10, 99.95, 100.05, 1000, 100],
            [100, 100.40, 100.00, 100.35, 1000, 100],
            [100, 100.42, 100.20, 100.30, 1000, 100],
        ])
        labeled = create_multiclass_labels(
            df,
            BarrierConfig(horizon_bars=2, target_pct=0.003, stop_pct=0.0015),
        )
        self.assertEqual(int(labeled.iloc[0]["trade_label"]), int(TradeLabel.LONG))
        self.assertEqual(labeled.iloc[0]["trade_label_name"], "LONG")

    def test_both_direction_wins_becomes_wait(self):
        df = frame([
            [100, 100.02, 99.98, 100, 1000, 100],
            [100, 100.40, 100.00, 100.30, 1000, 100],
            [100, 100.30, 99.60, 99.70, 1000, 100],
            [100, 100.10, 99.80, 100.00, 1000, 100],
        ])
        labeled = create_multiclass_labels(
            df,
            BarrierConfig(horizon_bars=2, target_pct=0.003, stop_pct=0.005),
        )
        self.assertEqual(int(labeled.iloc[0]["trade_label"]), int(TradeLabel.WAIT))

    def test_labels_do_not_cross_session_boundary(self):
        index = pd.to_datetime([
            "2026-01-02 15:58",
            "2026-01-02 15:59",
            "2026-01-05 09:30",
            "2026-01-05 09:31",
        ])
        df = frame([
            [100, 100.02, 99.98, 100, 1000, 100],
            [100, 100.05, 99.95, 100, 1000, 100],
            [110, 110.50, 109.80, 110.30, 1000, 110],
            [110, 110.60, 110.00, 110.40, 1000, 110],
        ])
        df.index = index
        labeled = create_multiclass_labels(
            df,
            BarrierConfig(horizon_bars=1, target_pct=0.003, stop_pct=0.0015),
        )
        self.assertFalse(bool(labeled.iloc[1]["label_valid"]))
        self.assertTrue(pd.isna(labeled.iloc[1]["trade_label"]))

    def test_atr_is_causal(self):
        df = frame([
            [100, 101, 99, 100, 1000, 100],
            [100, 102, 99, 101, 1000, 100],
            [101, 103, 100, 102, 1000, 101],
            [102, 104, 101, 103, 1000, 102],
        ])
        atr_before = calculate_atr(df, period=2)
        changed = df.copy()
        changed.loc[3, ["high", "low", "close"]] = [1000, 1, 500]
        atr_after = calculate_atr(changed, period=2)
        pd.testing.assert_series_equal(atr_before.iloc[:3], atr_after.iloc[:3])


if __name__ == "__main__":
    unittest.main()
