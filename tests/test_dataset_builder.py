import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.build_dataset import build_dataset


class FakeSource:
    source_name = "fake"

    def fetch_minute_bars(self, symbols, **kwargs):
        frames = []
        session_starts = pd.to_datetime([
            "2026-01-05 14:30Z",
            "2026-01-06 14:30Z",
            "2026-01-07 14:30Z",
        ])
        for symbol_number, symbol in enumerate(symbols):
            symbol_frames = []
            for session_number, start in enumerate(session_starts):
                index = pd.date_range(start, periods=90, freq="min")
                n = np.arange(len(index))
                base = 100.0 + symbol_number * 10 + session_number * 0.2
                wave = np.sin(n / 3.0) * 0.45
                close = base + wave
                symbol_frames.append(pd.DataFrame(
                    {
                        "symbol": symbol,
                        "open": close,
                        "high": close + 0.22,
                        "low": close - 0.22,
                        "close": close,
                        "volume": 1000 + session_number * 100 + n,
                        "vwap": close,
                        "trade_count": 100 + session_number * 10 + n,
                    },
                    index=index,
                ))
            frames.append(pd.concat(symbol_frames))
        return pd.concat(frames).sort_index()


class FakeStore:
    def __init__(self):
        self.raw = {}
        self.processed = {}
        self.manifest = None

    def write_raw_bars(self, version, symbol, df):
        self.raw[(version, symbol)] = df.copy()
        return [f"raw/{version}/{symbol}.parquet"]

    def write_processed_training(self, version, symbol, df):
        self.processed[(version, symbol)] = df.copy()
        return [f"processed/{version}/{symbol}.parquet"]

    def write_manifest(self, version, manifest):
        self.manifest = manifest
        return f"manifests/{version}.json"


class DatasetBuilderTests(unittest.TestCase):
    def test_builds_v2_manifest_context_and_processed_rows(self):
        store = FakeStore()
        manifest = build_dataset(
            version="v05-test",
            start=datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 1, 8, 0, 0, tzinfo=timezone.utc),
            symbols=["AAA", "BBB"],
            batch_size=4,
            source=FakeSource(),
            store=store,
        )

        self.assertEqual(manifest["dataset_version"], "v05-test")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["feature_engine"], "v2")
        self.assertEqual(manifest["symbols_succeeded"], ["AAA", "BBB"])
        self.assertEqual(manifest["context_symbols"], ["SPY", "QQQ"])
        self.assertEqual(set(manifest["acquisition_symbols"]), {"AAA", "BBB", "SPY", "QQQ"})
        self.assertGreater(manifest["row_counts"]["processed_total"], 0)
        self.assertEqual(
            sum(x["count"] for x in manifest["class_distribution"].values()),
            manifest["row_counts"]["processed_total"],
        )
        self.assertEqual(len(store.raw), 4)
        self.assertEqual(len(store.processed), 2)
        self.assertIs(store.manifest, manifest)


if __name__ == "__main__":
    unittest.main()
