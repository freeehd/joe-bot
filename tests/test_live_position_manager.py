import unittest
from datetime import datetime, timezone

from execution.broker import OrderIntent, OrderIntentType, OrderSide, OrderSnapshot, OrderStatus, PositionSnapshot
from risk.position_manager import LivePositionManager


UTC = timezone.utc


class LivePositionManagerTests(unittest.TestCase):
    def test_fill_then_broker_reconciliation(self):
        manager = LivePositionManager()
        intent = OrderIntent(
            "AAPL", OrderSide.BUY, 5, OrderIntentType.ENTRY, "entry-1", "test",
            metadata={"stop_price": 99.0, "target_price": 102.0, "sector": "Technology"},
        )
        fill = OrderSnapshot(
            "broker-1", "entry-1", "AAPL", OrderSide.BUY, 5, 5, OrderStatus.FILLED,
            100.0, updated_at=datetime(2026, 1, 5, 14, 31, tzinfo=UTC)
        )
        manager.apply_filled_order(intent, fill)
        self.assertEqual(manager.get("AAPL").direction, "LONG")
        self.assertTrue(manager.reconcile([PositionSnapshot("AAPL", 5, 100.0)])["consistent"])
        mismatch = manager.reconcile([PositionSnapshot("AAPL", 4, 100.0)])
        self.assertFalse(mismatch["consistent"])
        self.assertEqual(mismatch["mismatches"][0]["reason"], "quantity_mismatch")

    def test_exit_fill_closes_position(self):
        manager = LivePositionManager()
        entry = OrderIntent("AAPL", OrderSide.BUY, 2, OrderIntentType.ENTRY, "e", "test")
        manager.apply_filled_order(entry, OrderSnapshot("1", "e", "AAPL", OrderSide.BUY, 2, 2, OrderStatus.FILLED, 100.0))
        exit_intent = OrderIntent("AAPL", OrderSide.SELL, 2, OrderIntentType.EXIT, "x", "target")
        manager.apply_filled_order(exit_intent, OrderSnapshot("2", "x", "AAPL", OrderSide.SELL, 2, 2, OrderStatus.FILLED, 101.0))
        self.assertIsNone(manager.get("AAPL"))


if __name__ == "__main__":
    unittest.main()
