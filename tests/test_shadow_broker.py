import unittest
from datetime import datetime, timedelta, timezone

from backtest.engine import ExecutionConfig
from execution.broker import OrderIntent, OrderIntentType, OrderSide, OrderStatus
from execution.shadow_broker import ShadowBroker
from market.stream import BarEvent

UTC = timezone.utc


class ShadowBrokerTests(unittest.TestCase):
    def test_order_is_local_only_and_fills_next_bar(self):
        broker = ShadowBroker(initial_equity=10_000, execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0))
        t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        intent = OrderIntent("AAPL", OrderSide.BUY, 5, OrderIntentType.ENTRY, "shadow-1", "test", {"signal_time": t0.isoformat()})
        accepted = broker.submit_market_order(intent)
        self.assertEqual(accepted.status, OrderStatus.ACCEPTED)
        self.assertFalse(broker.external_orders)
        self.assertEqual(broker.on_market_event(BarEvent("AAPL", t0, 100, 100, 100, 100, 1000, received_at=t0)), [])
        t1 = t0 + timedelta(minutes=1)
        updates = broker.on_market_event(BarEvent("AAPL", t1, 101, 102, 100, 101.5, 1000, received_at=t1))
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].status, OrderStatus.FILLED)
        self.assertEqual(updates[0].average_fill_price, 101.0)
        self.assertEqual(broker.get_positions()[0].quantity, 5)


if __name__ == "__main__":
    unittest.main()
