import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database.db import AuditStore
from research.campaign_metrics import CampaignThresholds, evaluate_campaign_gate, summarize_paper_campaign, summarize_shadow_campaign
from research.fault_injection import run_fault_injection

UTC = timezone.utc


class CampaignMetricsTests(unittest.TestCase):
    def test_shadow_campaign_matches_expected_and_realized_by_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow.sqlite3"
            with AuditStore(path) as store:
                now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
                store.append("shadow_mode_started", {"external_orders": False}, timestamp=now)
                store.append("shadow_entry_filled", {"expected_ev_bps": 8.0, "fill_slippage_bps": 2.0, "signal_to_fill_seconds": 60.0}, decision_id="d1", timestamp=now)
                store.append("shadow_trade_closed", {"realized_bps": 6.0}, decision_id="d1", timestamp=now + timedelta(minutes=2))
                # Unmatched close must not contaminate EV drift.
                store.append("shadow_trade_closed", {"realized_bps": -99.0}, decision_id="unknown", timestamp=now + timedelta(minutes=3))
            result = summarize_shadow_campaign([path])
            self.assertEqual(result["sessions"], 1)
            self.assertEqual(result["closed_trades"], 2)
            self.assertEqual(result["matched_ev_trades"], 1)
            self.assertEqual(result["average_expected_ev_bps"], 8.0)
            self.assertEqual(result["average_realized_ev_bps"], 6.0)
            self.assertEqual(result["expected_vs_realized_delta_bps"], -2.0)
            self.assertEqual(result["clean_session_rate"], 1.0)

    def test_paper_campaign_detects_duplicate_and_reconciliation_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.sqlite3"
            with AuditStore(path) as store:
                now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
                store.append("engine_started", {}, timestamp=now)
                submit = {"intent": {"client_order_id": "exit-aapl-1"}, "broker_order": {"client_order_id": "exit-aapl-1"}}
                store.append("exit_order_submitted", submit, timestamp=now)
                store.append("exit_order_submitted", submit, timestamp=now + timedelta(seconds=1))
                store.append("order_update", {"client_order_id": "exit-aapl-1", "status": "FILLED"}, timestamp=now + timedelta(seconds=2))
                store.append("position_reconciliation", {"consistent": False}, timestamp=now + timedelta(seconds=3))
            result = summarize_paper_campaign([path])
            self.assertEqual(result["closed_trades"], 1)
            self.assertEqual(result["duplicate_order_submissions"], 1)
            self.assertEqual(result["reconciliation_mismatches"], 1)
            self.assertEqual(result["clean_session_rate"], 0.0)

    def test_campaign_gate_requires_every_mandatory_condition(self):
        shadow = {
            "sessions": 2, "closed_trades": 5, "average_realized_ev_bps": 3.0,
            "expected_vs_realized_delta_bps": -1.0, "clean_session_rate": 1.0, "runtime_crashes": 0,
        }
        paper = {
            "sessions": 2, "closed_trades": 5, "clean_session_rate": 1.0, "runtime_crashes": 0,
            "reconciliation_mismatches": 0, "duplicate_order_submissions": 0,
        }
        fault = {"pass_rate": 1.0}
        thresholds = CampaignThresholds(min_shadow_sessions=2, min_shadow_closed_trades=5, min_paper_sessions=2, min_paper_closed_trades=5)
        report = evaluate_campaign_gate(shadow, paper, fault, thresholds=thresholds)
        self.assertTrue(report["passed"])
        self.assertEqual(report["verdict"], "PASS")
        self.assertFalse(report["live_capital_authorized"])
        paper["reconciliation_mismatches"] = 1
        self.assertFalse(evaluate_campaign_gate(shadow, paper, fault, thresholds=thresholds)["passed"])

    def test_fault_injection_safety_invariants_all_pass(self):
        result = run_fault_injection()
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["pass_rate"], 1.0)
        self.assertFalse(result["external_orders"])
        self.assertGreaterEqual(result["total_cases"], 5)


if __name__ == "__main__":
    unittest.main()
