import unittest
from datetime import datetime, timedelta, timezone

from backtest.engine import ExecutionConfig
from database.db import AuditStore
from execution.exit_engine import ExitEngine, ExitPolicy
from execution.shadow_broker import ShadowBroker
from live.paper_engine import EntryProposal
from live.shadow_engine import ShadowTradingEngine
from market.stream import BarEvent
from risk.risk_engine import ProductionRiskEngine, RiskLimits

UTC = timezone.utc


class ShadowEngineTests(unittest.TestCase):
    def test_complete_shadow_trade_records_expected_and_realized_without_external_orders(self):
        t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        emitted = [False]
        def provider(event, state):
            if emitted[0]:
                return []
            emitted[0] = True
            return [EntryProposal("AAPL", "LONG", 2, event.close, 99.0, 101.0, 8.0, decision_id="decision-1", sector="Technology")]

        broker = ShadowBroker(initial_equity=10_000, execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0))
        audit = AuditStore(":memory:")
        engine = ShadowTradingEngine(
            broker=broker,
            risk_engine=ProductionRiskEngine(RiskLimits(stale_market_data_seconds=60, max_spread_bps=100)),
            exit_engine=ExitEngine(ExitPolicy(target_pct=0.01, stop_pct=0.01, max_holding_minutes=30)),
            audit_store=audit,
            candidate_provider=provider,
        )
        engine.start(now=t0)
        engine.risk_engine.update_health(broker_connected=True, websocket_stable=True, model_ready=True, features_valid=True, position_state_consistent=True)
        engine.on_market_event(BarEvent("AAPL", t0, 100, 100.2, 99.8, 100, 1000, received_at=t0))
        t1 = t0 + timedelta(minutes=1)
        engine.on_market_event(BarEvent("AAPL", t1, 100, 100.4, 99.9, 100, 1000, received_at=t1))
        self.assertIsNotNone(engine.positions.get("AAPL"))
        t2 = t1 + timedelta(minutes=1)
        engine.on_market_event(BarEvent("AAPL", t2, 101.1, 101.2, 101.0, 101.1, 1000, received_at=t2))
        t3 = t2 + timedelta(minutes=1)
        engine.on_market_event(BarEvent("AAPL", t3, 101.2, 101.3, 101.1, 101.2, 1000, received_at=t3))
        self.assertIsNone(engine.positions.get("AAPL"))
        self.assertFalse(broker.external_orders)
        types = [e.event_type for e in audit.iter_events()]
        self.assertIn("shadow_candidate", types)
        self.assertIn("shadow_entry_filled", types)
        self.assertIn("shadow_trade_closed", types)
        close = [e for e in audit.iter_events() if e.event_type == "shadow_trade_closed"][0]
        self.assertEqual(close.decision_id, "decision-1")
        self.assertGreater(close.payload["realized_bps"], 0)


if __name__ == "__main__":
    unittest.main()
