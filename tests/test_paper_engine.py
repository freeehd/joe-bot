import unittest
from datetime import datetime, timedelta, timezone

from database.db import AuditStore
from execution.broker import (
    AccountSnapshot,
    OrderIntentType,
    OrderSnapshot,
    OrderStatus,
    PositionSnapshot,
)
from execution.exit_engine import ExitEngine, ExitPolicy
from live.paper_engine import EntryProposal, PaperTradingEngine
from market.stream import BarEvent
from risk.risk_engine import ProductionRiskEngine, RiskLimits


UTC = timezone.utc


class FakePaperBroker:
    paper = True

    def __init__(self):
        self.account = AccountSnapshot(10_000, 10_000, 10_000)
        self.positions = []
        self.submissions = []
        self.open_orders = []

    def submit_market_order(self, intent):
        self.submissions.append(intent)
        snapshot = OrderSnapshot(
            broker_order_id=f"b{len(self.submissions)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            filled_quantity=0,
            status=OrderStatus.ACCEPTED,
            submitted_at=datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
        )
        self.open_orders.append(snapshot)
        return snapshot

    def cancel_order(self, broker_order_id):
        pass

    def get_open_orders(self):
        return list(self.open_orders)

    def get_positions(self):
        return list(self.positions)

    def get_account(self):
        return self.account

    def fill_last(self, price):
        intent = self.submissions[-1]
        accepted = self.open_orders[-1]
        filled = OrderSnapshot(
            accepted.broker_order_id,
            accepted.client_order_id,
            accepted.symbol,
            accepted.side,
            accepted.quantity,
            accepted.quantity,
            OrderStatus.FILLED,
            price,
            submitted_at=accepted.submitted_at,
            updated_at=accepted.submitted_at + timedelta(seconds=1),
        )
        self.open_orders[-1] = filled
        signed = filled.quantity if filled.side.value == "BUY" else -filled.quantity
        if intent.intent_type == OrderIntentType.ENTRY:
            self.positions = [PositionSnapshot(filled.symbol, signed, price, market_price=price)]
        else:
            self.positions = []
        return filled


class PaperEngineTests(unittest.TestCase):
    def _engine(self, broker, provider=None):
        risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=60, max_spread_bps=100))
        return PaperTradingEngine(
            broker=broker,
            risk_engine=risk,
            exit_engine=ExitEngine(ExitPolicy(target_pct=0.01, stop_pct=0.01, max_holding_minutes=30)),
            audit_store=AuditStore(":memory:"),
            candidate_provider=provider,
        )

    def test_entry_fill_reconciles_and_target_exit_is_submitted(self):
        broker = FakePaperBroker()
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)

        def provider(event, state):
            return [EntryProposal("AAPL", "LONG", 5, event.close, 99.0, 101.0, 8.0, decision_id="d1", sector="Technology")]

        engine = self._engine(broker, provider)
        engine.start(now=now)
        engine.on_market_event(BarEvent("AAPL", now, 100, 100.1, 99.9, 100, 1000, received_at=now))
        self.assertEqual(len(broker.submissions), 1)
        engine.on_order_update(broker.fill_last(100.0))
        self.assertIsNotNone(engine.positions.get("AAPL"))

        # Repeated strategy proposal must not create a second entry while held;
        # the target event should create exactly one EXIT instead.
        later = now + timedelta(minutes=1)
        engine.on_market_event(BarEvent("AAPL", later, 101.0, 101.2, 100.9, 101.1, 1000, received_at=later))
        self.assertEqual(len(broker.submissions), 2)
        self.assertEqual(broker.submissions[-1].intent_type, OrderIntentType.EXIT)

    def test_manual_entry_halt_still_allows_exit(self):
        broker = FakePaperBroker()
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        engine = self._engine(broker)
        broker.positions = [PositionSnapshot("AAPL", 2, 100.0, market_price=100.0)]
        engine.start(now=now)
        # Adopted broker positions do not have levels, so ExitEngine derives them.
        engine.risk_engine.manual_halt("operator")
        later = now + timedelta(minutes=1)
        engine.on_market_event(BarEvent("AAPL", later, 101, 101.2, 100.9, 101.1, 1000, received_at=later))
        self.assertEqual(len(broker.submissions), 1)
        self.assertEqual(broker.submissions[0].intent_type, OrderIntentType.EXIT)

    def test_position_mismatch_activates_kill_switch(self):
        broker = FakePaperBroker()
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        broker.positions = [PositionSnapshot("AAPL", 1, 100.0)]
        engine = self._engine(broker)
        engine.start(now=now)
        broker.positions = [PositionSnapshot("AAPL", 2, 100.0)]
        result = engine.reconcile_positions(timestamp=now)
        self.assertFalse(result["consistent"])
        self.assertIn("position_state_inconsistent", engine.state(now).kill_switches)


if __name__ == "__main__":
    unittest.main()
