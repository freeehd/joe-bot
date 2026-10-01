import unittest
from datetime import datetime, timezone

from database.db import AuditStore
from live.shadow_report import summarize_shadow

UTC = timezone.utc


class ShadowReportTests(unittest.TestCase):
    def test_summary_compares_expected_and_realized(self):
        store = AuditStore(":memory:")
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        store.append("shadow_candidate", {"proposal": {}}, decision_id="d1", timestamp=now)
        store.append("risk_decision", {"risk": {"approved": True}}, decision_id="d1", timestamp=now)
        store.append("shadow_entry_filled", {"expected_ev_bps": 8.0, "fill_slippage_bps": 2.0, "signal_to_fill_seconds": 60.0}, decision_id="d1", timestamp=now)
        store.append("shadow_trade_closed", {"realized_bps": 12.0}, decision_id="d1", timestamp=now)
        result = summarize_shadow(store)
        self.assertEqual(result["closed_trades"], 1)
        self.assertEqual(result["average_expected_ev_bps"], 8.0)
        self.assertEqual(result["average_realized_bps"], 12.0)
        self.assertEqual(result["expected_vs_realized_delta_bps"], 4.0)
        self.assertEqual(result["average_signal_to_fill_seconds"], 60.0)
        self.assertEqual(result["runtime_crashes"], 0)


if __name__ == "__main__":
    unittest.main()
