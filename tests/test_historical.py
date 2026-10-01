import unittest

import pandas as pd

from market.historical import filter_regular_hours, normalize_bar_frame


class HistoricalNormalizationTests(unittest.TestCase):
    def test_normalizes_multiindex_alpaca_shape(self):
        index = pd.MultiIndex.from_arrays(
            [
                ["AAPL", "AAPL"],
                pd.to_datetime(["2026-01-02 14:30Z", "2026-01-02 14:31Z"]),
            ],
            names=["symbol", "timestamp"],
        )
        raw = pd.DataFrame(
            {
                "open": [100.0, 100.1],
                "high": [100.2, 100.3],
                "low": [99.9, 100.0],
                "close": [100.1, 100.2],
                "volume": [1000, 1100],
                "vwap": [100.05, 100.15],
            },
            index=index,
        )
        normalized = normalize_bar_frame(raw)
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        self.assertEqual(str(normalized.index.tz), "UTC")
        self.assertEqual(normalized["symbol"].tolist(), ["AAPL", "AAPL"])

    def test_regular_hours_filter_uses_new_york_time(self):
        index = pd.to_datetime([
            "2026-07-01 13:29Z",  # 09:29 ET
            "2026-07-01 13:30Z",  # 09:30 ET
            "2026-07-01 19:59Z",  # 15:59 ET
            "2026-07-01 20:00Z",  # 16:00 ET
        ])
        frame = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "open": [1.0] * 4,
                "high": [1.0] * 4,
                "low": [1.0] * 4,
                "close": [1.0] * 4,
                "volume": [1] * 4,
                "vwap": [1.0] * 4,
            },
            index=index,
        )
        result = filter_regular_hours(frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.index[0], index[1])
        self.assertEqual(result.index[-1], index[2])


if __name__ == "__main__":
    unittest.main()
