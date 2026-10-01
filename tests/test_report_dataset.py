import unittest

from research.report_dataset import render_manifest


class DatasetReportTests(unittest.TestCase):
    def test_report_contains_core_dataset_information(self):
        manifest = {
            "dataset_version": "v04-r50",
            "source": {"provider": "alpaca", "feed": "iex"},
            "date_range": {"start": "2024-01-01", "end": "2026-01-01"},
            "symbols_requested": ["A", "B"],
            "symbols_succeeded": ["A"],
            "features": ["f1", "f2"],
            "feature_engine": "v2",
            "context_symbols": ["SPY", "QQQ"],
            "failed_symbols": {"B": "no data"},
            "row_counts": {"raw_total": 100, "processed_total": 80},
            "class_distribution": {
                "WAIT": {"count": 40, "percent": 50.0},
                "LONG": {"count": 20, "percent": 25.0},
                "SHORT": {"count": 20, "percent": 25.0},
            },
            "quality": {
                "A": {
                    "missing_minute_intervals": 3,
                    "duplicate_rows": 1,
                    "invalid_numeric_rows": 1,
                    "invalid_ohlc_rows": 0,
                }
            },
        }
        text = render_manifest(manifest)
        self.assertIn("v04-r50", text)
        self.assertIn("Processed rows: 80", text)
        self.assertIn("Feature engine: v2 (2 features)", text)
        self.assertIn("Context symbols: SPY, QQQ", text)
        self.assertIn("B: no data", text)


if __name__ == "__main__":
    unittest.main()
