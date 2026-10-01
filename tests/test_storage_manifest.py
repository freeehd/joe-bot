import tempfile
import unittest
from pathlib import Path

from research.storage import ParquetDataLake


class StorageManifestTests(unittest.TestCase):
    def test_partition_layout(self):
        lake = ParquetDataLake("data")
        self.assertEqual(
            lake.raw_partition_path("v04-test", "aapl", 2026, 1),
            Path("data/raw/datasets/v04-test/bars/symbol=AAPL/year=2026/month=01/bars.parquet"),
        )
        self.assertEqual(
            lake.processed_partition_path("v04-test", "aapl", 2026, 1),
            Path(
                "data/processed/datasets/v04-test/"
                "symbol=AAPL/year=2026/month=01/training.parquet"
            ),
        )

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lake = ParquetDataLake(tmp)
            payload = {"dataset_version": "v04-test", "row_counts": {"processed_total": 42}}
            path = lake.write_manifest("v04-test", payload)
            self.assertTrue(path.exists())
            self.assertEqual(lake.read_manifest("v04-test"), payload)

    def test_dataset_versions_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            lake = ParquetDataLake(tmp)
            lake.write_manifest("v04-test", {"dataset_version": "v04-test"})
            with self.assertRaises(FileExistsError):
                lake.assert_new_version("v04-test")


if __name__ == "__main__":
    unittest.main()
