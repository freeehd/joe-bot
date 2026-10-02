import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ops.recovery import RecoveryStore
from ops.supervisor import RuntimeSupervisor

UTC = timezone.utc


class RecoveryHardeningTests(unittest.TestCase):
    def test_atomic_snapshot_round_trip_and_checksum_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            store = RecoveryStore(path)
            saved = store.save({"positions": [{"symbol": "AAPL", "qty": 2}]}, sequence=7)
            loaded = store.load()
            self.assertEqual(loaded.sequence, 7)
            self.assertEqual(loaded.checksum_sha256, saved.checksum_sha256)
            raw = json.loads(path.read_text())
            raw["state"]["positions"][0]["qty"] = 999
            path.write_text(json.dumps(raw))
            with self.assertRaises(RuntimeError):
                store.load()

    def test_supervisor_enforces_heartbeat_and_restart_budget(self):
        now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
        sup = RuntimeSupervisor(heartbeat_timeout_seconds=5, max_restarts_per_hour=2)
        sup.heartbeat("market", now=now)
        self.assertTrue(sup.evaluate(now=now + timedelta(seconds=4))["healthy"])
        self.assertFalse(sup.evaluate(now=now + timedelta(seconds=6))["healthy"])
        sup.record_restart("market", now=now)
        sup.record_restart("market", now=now + timedelta(minutes=1))
        self.assertFalse(sup.can_restart("market", now=now + timedelta(minutes=2)))
        with self.assertRaises(RuntimeError):
            sup.record_restart("market", now=now + timedelta(minutes=2))


if __name__ == "__main__":
    unittest.main()
