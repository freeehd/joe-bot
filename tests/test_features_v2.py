import unittest

import numpy as np
import pandas as pd

from features.v2 import FEATURE_COLUMNS_V2, add_market_context, create_features_v2


def make_symbol(symbol: str, start: str, sessions: int = 3, bars_per_session: int = 70):
    frames = []
    base_day = pd.Timestamp(start, tz="America/New_York")
    for day in range(sessions):
        session_start = base_day + pd.Timedelta(days=day)
        idx_local = pd.date_range(session_start, periods=bars_per_session, freq="min")
        idx = idx_local.tz_convert("UTC")
        n = np.arange(bars_per_session)
        close = 100 + day + n * 0.01 + np.sin(n / 5) * 0.05
        frames.append(pd.DataFrame({
            "symbol": symbol,
            "open": close - 0.01,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": 1000 + day * 100 + n * 3,
            "vwap": close - 0.02,
            "trade_count": 100 + day * 10 + n,
        }, index=idx))
    return pd.concat(frames).sort_index()


class FeatureEngineV2Tests(unittest.TestCase):
    def test_feature_schema_is_in_target_range_and_unique(self):
        self.assertGreaterEqual(len(FEATURE_COLUMNS_V2), 30)
        self.assertLessEqual(len(FEATURE_COLUMNS_V2), 80)
        self.assertEqual(len(FEATURE_COLUMNS_V2), len(set(FEATURE_COLUMNS_V2)))

    def test_returns_reset_at_new_york_session_boundary(self):
        raw = make_symbol("AAA", "2026-01-05 09:30", sessions=2)
        features = create_features_v2(raw)
        local_dates = features.index.tz_convert("America/New_York").date
        second_date = sorted(set(local_dates))[1]
        first_second_session = features.loc[local_dates == second_date].iloc[0]
        self.assertTrue(pd.isna(first_second_session["return_1m"]))
        self.assertTrue(pd.isna(first_second_session["return_15m"]))

    def test_time_of_day_relative_volume_is_historical_only(self):
        raw = make_symbol("AAA", "2026-01-05 09:30", sessions=3)
        features = create_features_v2(raw)
        local = features.index.tz_convert("America/New_York")
        open_rows = features.loc[(local.hour == 9) & (local.minute == 30)]
        self.assertEqual(len(open_rows), 3)
        self.assertTrue(pd.isna(open_rows.iloc[0]["tod_relative_volume"]))
        expected_second = open_rows.iloc[1]["volume"] / open_rows.iloc[0]["volume"]
        self.assertAlmostEqual(open_rows.iloc[1]["tod_relative_volume"], expected_second)
        expected_third_baseline = (open_rows.iloc[0]["volume"] + open_rows.iloc[1]["volume"]) / 2
        self.assertAlmostEqual(
            open_rows.iloc[2]["tod_relative_volume"],
            open_rows.iloc[2]["volume"] / expected_third_baseline,
        )

    def test_market_context_aligns_benchmarks_and_breadth(self):
        frames = []
        for i, symbol in enumerate(["SPY", "QQQ", "AAA"]):
            raw = make_symbol(symbol, "2026-01-05 09:30", sessions=2)
            raw["close"] = raw["close"] * (1 + i * 0.0001)
            features = create_features_v2(raw)
            features["training_symbol"] = symbol
            frames.append(features)
        dataset = pd.concat(frames).sort_index()
        context = add_market_context(dataset)
        usable = context.dropna(subset=["return_5m", "spy_return_5m", "qqq_return_5m"])
        self.assertGreater(len(usable), 0)
        aaa = usable.loc[usable["training_symbol"] == "AAA"].iloc[-1]
        self.assertAlmostEqual(
            aaa["relative_strength_spy_5m"],
            aaa["return_5m"] - aaa["spy_return_5m"],
        )
        self.assertGreaterEqual(aaa["breadth_up_1m"], 0.0)
        self.assertLessEqual(aaa["breadth_up_1m"], 1.0)


if __name__ == "__main__":
    unittest.main()
