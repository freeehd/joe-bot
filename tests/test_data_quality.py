import unittest

import pandas as pd

from research.quality import clean_and_validate_bars


class DataQualityTests(unittest.TestCase):
    def test_duplicates_invalid_rows_and_gaps_are_explicit(self):
        index = pd.to_datetime([
            "2026-01-02 14:30Z",
            "2026-01-02 14:31Z",
            "2026-01-02 14:31Z",  # duplicate
            "2026-01-02 14:33Z",  # 14:32 missing
            "2026-01-02 14:34Z",  # invalid OHLC
            "2026-01-02 14:35Z",  # negative volume
        ])
        frame = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 6,
                "open": [100, 100, 100, 100, 100, 100],
                "high": [101, 101, 101.1, 101, 99, 101],
                "low": [99, 99, 99, 99, 98, 99],
                "close": [100, 100, 100.1, 100, 100, 100],
                "volume": [1000, 1000, 1100, 1000, 1000, -1],
                "vwap": [100] * 6,
            },
            index=index,
        )
        cleaned, report = clean_and_validate_bars(frame, symbol="AAPL")
        self.assertEqual(report.input_rows, 6)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.invalid_ohlc_rows, 1)
        self.assertEqual(report.invalid_numeric_rows, 1)
        self.assertGreaterEqual(report.missing_minute_intervals, 1)
        self.assertEqual(len(cleaned), 3)


if __name__ == "__main__":
    unittest.main()
