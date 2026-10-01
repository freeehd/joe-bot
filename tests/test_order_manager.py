import unittest
from datetime import datetime, timezone

from execution.broker import AccountSnapshot, OrderIntent, OrderIntentType, OrderSide, OrderSnapshot, OrderStatus
from execution.order_manager import OrderLifecycleError, OrderManager


UTC = timezone.utc


class FakeBroker:
    paper = True

    def __init__(self):
        self.submissions = []
        self.orders = []

    def submit_market_order(self, intent):
        self.submissions.append(intent)
        snapshot = OrderSnapshot(
            broker_order_id=f"broker-{len(self.submissions)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            filled_quantity=0,
            status=OrderStatus.ACCEPTED,
            submitted_at=datetime.now(tz=UTC),
        )
        self.orders.append(snapshot)
        return snapshot

    def cancel_order(self, broker_order_id):
        pass

    def get_open_orders(self):
        return list(self.orders)

    def get_positions(self):
        return []

    def get_account(self):
        return AccountSnapshot(10_000, 10_000, 10_000)


class OrderManagerTests(unittest.TestCase):
    def test_client_order_id_makes_submission_idempotent(self):
        broker = FakeBroker()
        manager = OrderManager(broker)
        intent = OrderIntent("AAPL", OrderSide.BUY, 3, OrderIntentType.ENTRY, "abc", "test")
        first = manager.submit(intent)
        second = manager.submit(intent)
        self.assertEqual(first, second)
        self.assertEqual(len(broker.submissions), 1)

    def test_invalid_terminal_transition_is_rejected(self):
        broker = FakeBroker()
        manager = OrderManager(broker)
        intent = OrderIntent("AAPL", OrderSide.BUY, 3, OrderIntentType.ENTRY, "abc", "test")
        accepted = manager.submit(intent)
        filled = OrderSnapshot(
            accepted.broker_order_id, "abc", "AAPL", OrderSide.BUY, 3, 3,
            OrderStatus.FILLED, 100.0, updated_at=datetime.now(tz=UTC)
        )
        manager.apply_update(filled)
        canceled = OrderSnapshot(
            accepted.broker_order_id, "abc", "AAPL", OrderSide.BUY, 3, 3,
            OrderStatus.CANCELED, 100.0, updated_at=datetime.now(tz=UTC)
        )
        with self.assertRaises(OrderLifecycleError):
            manager.apply_update(canceled)


if __name__ == "__main__":
    unittest.main()
