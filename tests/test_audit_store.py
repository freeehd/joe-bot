import tempfile
import unittest
from pathlib import Path

from database.db import AuditStore


class AuditStoreTests(unittest.TestCase):
    def test_append_only_decision_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore(Path(directory) / "audit.sqlite3")
            first = store.append("decision", {"ev": 8.2}, decision_id="d1", symbol="aapl")
            store.append("risk_decision", {"approved": True}, decision_id="d1", symbol="AAPL")
            replay = store.replay("d1")
            self.assertIsNotNone(first.event_id)
            self.assertEqual(replay["event_count"], 2)
            self.assertEqual(replay["events"][0]["symbol"], "AAPL")
            self.assertEqual(replay["events"][1]["event_type"], "risk_decision")
            store.close()


if __name__ == "__main__":
    unittest.main()
