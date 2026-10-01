import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.build_dataset import build_dataset


class FakeSource:
    source_name = "fake"

    def fetch_minute_bars(self, symbols, **kwargs):
        frames = []
        index = pd.date_range("2026-01-05 14:30Z", periods=90, freq="min")
        for symbol_number, symbol in enumerate(symbols):
            base = 100.0 + symbol_number * 10
            wave = np.sin(np.arange(len(index)) / 3.0) * 0.45
            close = base + wave
            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "open": close,
                    "high": close + 0.22,
                    "low": close - 0.22,
                    "close": close,
                    "volume": 1000 + np.arange(len(index)),
                    "vwap": close,
                },
                index=index,
            )
            frames.append(frame)
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
    def test_builds_versioned_manifest_and_processed_rows(self):
        store = FakeStore()
        manifest = build_dataset(
            version="v04-test",
            start=datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 1, 6, 0, 0, tzinfo=timezone.utc),
            symbols=["AAA", "BBB"],
            batch_size=2,
            source=FakeSource(),
            store=store,
        )

        self.assertEqual(manifest["dataset_version"], "v04-test")
        self.assertEqual(manifest["symbols_succeeded"], ["AAA", "BBB"])
        self.assertGreater(manifest["row_counts"]["processed_total"], 0)
        self.assertEqual(sum(x["count"] for x in manifest["class_distribution"].values()),
                         manifest["row_counts"]["processed_total"])
        self.assertEqual(len(store.raw), 2)
        self.assertEqual(len(store.processed), 2)
        self.assertIs(store.manifest, manifest)


if __name__ == "__main__":
    unittest.main()
