import sqlite3
import tempfile
import unittest
from pathlib import Path

from ops.backup import backup_sqlite, restore_sqlite
from ops.failover import DataSourceState, MarketDataFailover
from ops.slo import SLOThresholds, evaluate_slo


class BackupFailoverSloTests(unittest.TestCase):
    def test_sqlite_backup_restore_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite3"
            with sqlite3.connect(source) as conn:
                conn.execute("create table t(x integer)")
                conn.execute("insert into t values (42)")
            result = backup_sqlite(source, root / "backup.sqlite3")
            restored = restore_sqlite(result.backup, root / "restored.sqlite3", expected_sha256=result.sha256)
            with sqlite3.connect(restored) as conn:
                self.assertEqual(conn.execute("select x from t").fetchone()[0], 42)

    def test_failover_prefers_first_healthy_source(self):
        policy = MarketDataFailover(["primary", "secondary"], max_latency_ms=100)
        chosen = policy.choose([
            DataSourceState("primary", True, True, 20),
            DataSourceState("secondary", True, False, 50),
        ])
        self.assertEqual(chosen, "secondary")
        self.assertIsNone(policy.choose([DataSourceState("primary", False, False, 10)]))

    def test_slo_gate_requires_availability_latency_and_error_rate(self):
        good = evaluate_slo(total_checks=1000, successful_checks=1000, errors=0, latencies_ms=[10.0] * 1000)
        self.assertTrue(good["passed"])
        bad = evaluate_slo(
            total_checks=1000, successful_checks=990, errors=10, latencies_ms=[2000.0] * 1000,
            thresholds=SLOThresholds(min_availability=0.999, max_p95_latency_ms=100, max_error_rate=0.001),
        )
        self.assertFalse(bad["passed"])


if __name__ == "__main__":
    unittest.main()
