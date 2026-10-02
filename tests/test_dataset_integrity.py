import json
import tempfile
import unittest
from pathlib import Path

from research.dataset_integrity import audit_dataset, freeze_dataset
from research.storage import ParquetDataLake


class DatasetIntegrityTests(unittest.TestCase):
    def _write_fixture(self, root: Path):
        data = root / "data"
        raw = data / "raw" / "datasets" / "v1" / "bars" / "symbol=AAPL" / "year=2026" / "month=01" / "bars.parquet"
        processed = data / "processed" / "datasets" / "v1" / "symbol=AAPL" / "year=2026" / "month=01" / "training.parquet"
        raw.parent.mkdir(parents=True)
        processed.parent.mkdir(parents=True)
        raw.write_bytes(b"raw")
        processed.write_bytes(b"processed")
        manifest = {
            "dataset_version": "v1",
            "symbols_requested": [f"S{i}" for i in range(50)],
            "symbols_succeeded": [f"S{i}" for i in range(50)],
            "context_symbols": ["SPY", "QQQ"],
            "features": ["f1"],
            "failed_symbols": {},
            "row_counts": {"raw_total": 1000, "processed_total": 800},
            "quality": {"AAPL": {"missing_minute_intervals": 1, "first_timestamp": "2024-01-01T00:00:00+00:00", "last_timestamp": "2026-01-02T00:00:00+00:00"}},
            "files": {"raw_partitions": [str(raw)], "processed_partitions": [str(processed)]},
        }
        ParquetDataLake(data).write_manifest("v1", manifest)
        return data, raw

    def test_freeze_is_repeatable_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, raw = self._write_fixture(Path(tmp))
            first = freeze_dataset("v1", root=data)
            second = freeze_dataset("v1", root=data)
            self.assertEqual(first["dataset_fingerprint"], second["dataset_fingerprint"])
            self.assertTrue(first["all_mandatory_gates_pass"])
            raw.write_bytes(b"mutated")
            with self.assertRaises(RuntimeError):
                freeze_dataset("v1", root=data)

    def test_audit_reports_missing_declared_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, raw = self._write_fixture(Path(tmp))
            raw.unlink()
            report = audit_dataset("v1", root=data)
            self.assertEqual(len(report["files_missing"]), 1)
            self.assertFalse(report["all_mandatory_gates_pass"])


if __name__ == "__main__":
    unittest.main()
