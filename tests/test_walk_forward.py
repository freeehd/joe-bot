import unittest

import pandas as pd

from research.walk_forward import (
    WalkForwardConfig,
    generate_windows,
    slice_window,
)


class WalkForwardTests(unittest.TestCase):
    def test_generates_rolling_monthly_windows(self):
        index = pd.date_range("2025-01-01", "2026-01-01", freq="D", inclusive="left", tz="UTC")
        config = WalkForwardConfig(
            train_months=6,
            calibration_months=1,
            test_months=1,
            step_months=1,
        )
        windows = generate_windows(index, config)
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0].train_start, pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertEqual(windows[0].test_start, pd.Timestamp("2025-08-01", tz="UTC"))
        self.assertEqual(windows[1].train_start, pd.Timestamp("2025-02-01", tz="UTC"))

    def test_slice_purges_train_and_calibration_tails(self):
        index = pd.date_range("2025-01-01", "2025-05-01", freq="D", inclusive="left", tz="UTC")
        dataset = pd.DataFrame({"trade_label": [0] * len(index)}, index=index)
        config = WalkForwardConfig(train_months=2, calibration_months=1, test_months=1)
        window = generate_windows(index, config)[0]
        train, calibration, test = slice_window(dataset, window, purge_bars=3)

        unpurged_train = dataset.loc[
            (dataset.index >= window.train_start) & (dataset.index < window.train_end)
        ]
        unpurged_cal = dataset.loc[
            (dataset.index >= window.calibration_start) & (dataset.index < window.calibration_end)
        ]
        self.assertEqual(len(train), len(unpurged_train) - 3)
        self.assertEqual(len(calibration), len(unpurged_cal) - 3)
        self.assertGreater(len(test), 0)
        self.assertLess(train.index.max(), window.train_end)
        self.assertLess(calibration.index.max(), window.calibration_end)


if __name__ == "__main__":
    unittest.main()


class _FakeLake:
    def __init__(self, raw_by_symbol):
        self.raw_by_symbol = raw_by_symbol

    def load_raw_bars(self, version, symbol):
        return self.raw_by_symbol[symbol]


class WalkForwardEndToEndTests(unittest.TestCase):
    def test_hist_gradient_boosting_runs_train_calibrate_test_and_execution(self):
        import numpy as np

        from backtest.engine import ExecutionConfig
        from research.walk_forward import run_walk_forward

        symbols = ["AAA", "BBB", "CCC"]
        rows = []
        raw_by_symbol = {}
        business_days = pd.bdate_range("2025-01-01", "2025-06-30", tz="UTC")

        for symbol_index, symbol in enumerate(symbols):
            raw_rows = []
            raw_index = []
            for day_index, day in enumerate(business_days):
                signal_time = day + pd.Timedelta(hours=14, minutes=30)
                next_time = signal_time + pd.Timedelta(minutes=1)
                feature = np.sin(day_index / 5.0) + symbol_index * 0.1
                label = symbol_index  # every timestamp has WAIT/LONG/SHORT represented
                rows.append(
                    {
                        "timestamp": signal_time,
                        "training_symbol": symbol,
                        "feature_1": feature,
                        "trade_label": label,
                    }
                )
                price = 100.0 + symbol_index
                raw_index.extend([signal_time, next_time])
                raw_rows.extend(
                    [
                        [price, price + 0.05, price - 0.05, price],
                        [price, price + 0.50, price - 0.50, price],
                    ]
                )
            raw_by_symbol[symbol] = pd.DataFrame(
                raw_rows,
                columns=["open", "high", "low", "close"],
                index=pd.DatetimeIndex(raw_index),
            )

        dataset = pd.DataFrame(rows).set_index("timestamp").sort_index()
        manifest = {
            "dataset_version": "synthetic-wf",
            "raw_snapshot_version": "synthetic-wf",
            "feature_engine": "v2",
            "label_parameters": {
                "horizon_bars": 1,
                "target_pct": 0.003,
                "stop_pct": 0.003,
                "use_atr": False,
                "atr_period": 14,
                "atr_target_multiplier": 1.0,
                "atr_stop_multiplier": 0.5,
            },
        }
        result = run_walk_forward(
            dataset,
            manifest,
            ["feature_1"],
            lake=_FakeLake(raw_by_symbol),
            model_name="hist_gradient_boosting",
            config=WalkForwardConfig(
                train_months=2,
                calibration_months=1,
                test_months=1,
                step_months=1,
                confidence_threshold=0.0,
            ),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
            stress_execution=True,
        )
        self.assertGreaterEqual(result["windows_completed"], 2)
        self.assertIn("combined_trade_metrics", result)
        self.assertIn("combined_adverse", result["execution_stress"])
