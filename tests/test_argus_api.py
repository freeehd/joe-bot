import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from argus_api.app import create_app
from argus_api.controls import PaperRiskControls
from argus_api.projector import AuditProjector
from argus_api.state import OperatorStateStore
from database.db import AuditStore
from risk.risk_engine import ProductionRiskEngine, RiskLimits

UTC = timezone.utc


class ArgusApiTests(unittest.TestCase):
    def test_default_api_is_read_only_and_live_capital_locked(self):
        state = OperatorStateStore()
        client = TestClient(create_app(state=state))
        config = client.get("/api/config").json()
        self.assertEqual(config["controls"], "read-only")
        self.assertFalse(config["live_capital_authorized"])
        self.assertEqual(client.post("/api/risk/disable-entries", json={"reason": "operator test"}).status_code, 403)

    def test_command_center_returns_projected_state(self):
        state = OperatorStateStore()
        state.update(mode="SHADOW", account={"equity": 10000, "cash": 9000, "buying_power": 9000})
        payload = TestClient(create_app(state=state)).get("/api/command-center").json()
        self.assertEqual(payload["mode"], "SHADOW")
        self.assertEqual(payload["account"]["equity"], 10000)

    def test_audit_projector_surfaces_runtime_health_and_alerts(self):
        state = OperatorStateStore()
        projector = AuditProjector(state)
        with AuditStore(":memory:") as audit:
            now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
            start = audit.append("engine_started", {"account": {"equity": 10000}}, timestamp=now)
            health = audit.append("runtime_health", {"kill_switches": ["market_data_stale"], "market_stream": {"connected": True, "stale": True, "last_event_at": now.isoformat()}, "trade_stream_connected": True}, timestamp=now)
            mismatch = audit.append("position_reconciliation", {"consistent": False}, timestamp=now)
            for event in (start, health, mismatch): projector.project_event(event)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["mode"], "PAPER")
        self.assertFalse(snapshot["risk"]["entries_enabled"])
        self.assertIn("market_data_stale", snapshot["risk"]["kill_switches"])
        self.assertEqual(snapshot["alerts"][0]["type"], "position_mismatch")

    def test_paper_risk_controls_are_audited_and_only_toggle_entries(self):
        risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=30))
        risk.reset_session(10000)
        with AuditStore(":memory:") as audit:
            controls = PaperRiskControls(risk, audit)
            self.assertTrue(controls.disable_entries("operator safety halt")["accepted"])
            self.assertTrue(risk.approve_exit().approved)
            self.assertTrue(controls.enable_entries("review completed")["accepted"])
            events = list(audit.iter_events(event_type="operator_control"))
            self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
