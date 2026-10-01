import unittest
from datetime import timezone

import pandas as pd

from features.v2 import FEATURE_COLUMNS_V2
from live.feature_store import LiveFeatureStore
from market.stream import BarEvent

UTC = timezone.utc


def bars_for_day(day: str, base: float) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 14:30:00", periods=70, freq="min", tz="UTC")
    rows = []
    for i, _ in enumerate(idx):
        price = base + i * 0.02
        rows.append({
            "open": price,
            "high": price + 0.05,
            "low": price - 0.05,
            "close": price + 0.01,
            "volume": 1000 + i * 3,
            "trade_count": 100 + i,
            "vwap": price,
        })
    return pd.DataFrame(rows, index=idx)


class LiveFeatureStoreTests(unittest.TestCase):
    def test_waits_for_synchronized_batch_and_reuses_v2_schema(self):
        symbols = ["AAA", "SPY", "QQQ"]
        history = {
            "AAA": bars_for_day("2026-01-02", 100),
            "SPY": bars_for_day("2026-01-02", 500),
            "QQQ": bars_for_day("2026-01-02", 450),
        }
        store = LiveFeatureStore(symbols, history, warmup_sessions=2)
        live = {
            "AAA": bars_for_day("2026-01-05", 101),
            "SPY": bars_for_day("2026-01-05", 501),
            "QQQ": bars_for_day("2026-01-05", 451),
        }
        last = None
        for i in range(70):
            ts = live["AAA"].index[i]
            for symbol in symbols:
                row = live[symbol].iloc[i]
                result = store.add_bar(BarEvent(
                    symbol, ts.to_pydatetime(), row.open, row.high, row.low, row.close,
                    row.volume, trade_count=row.trade_count, vwap=row.vwap, received_at=ts.to_pydatetime(),
                ))
                if symbol != "QQQ":
                    self.assertIsNone(result)
                else:
                    last = result
        self.assertIsNotNone(last)
        self.assertEqual(last.coverage, 1.0)
        self.assertEqual(set(last.frame["training_symbol"]), set(symbols))
        self.assertTrue(set(FEATURE_COLUMNS_V2).issubset(last.frame.columns))
        self.assertFalse(last.frame[FEATURE_COLUMNS_V2].isna().any().any())


if __name__ == "__main__":
    unittest.main()
