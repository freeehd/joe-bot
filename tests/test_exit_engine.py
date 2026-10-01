import unittest
from datetime import datetime, timedelta, timezone

from execution.exit_engine import ExitEngine, ExitPolicy, ExitReason
from risk.position_manager import ManagedPosition


UTC = timezone.utc


class ExitEngineTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
        self.engine = ExitEngine(ExitPolicy(target_pct=0.01, stop_pct=0.005, max_holding_minutes=10))

    def test_long_target_and_short_stop(self):
        long = ManagedPosition("AAPL", 2, 100.0, self.now)
        self.assertEqual(self.engine.evaluate(long, price=101.1, now=self.now).reason, ExitReason.TARGET)
        short = ManagedPosition("MSFT", -2, 100.0, self.now)
        self.assertEqual(self.engine.evaluate(short, price=100.6, now=self.now).reason, ExitReason.STOP)

    def test_time_exit(self):
        position = ManagedPosition("AAPL", 2, 100.0, self.now)
        decision = self.engine.evaluate(position, price=100.1, now=self.now + timedelta(minutes=11))
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, ExitReason.MAX_HOLDING_TIME)


if __name__ == "__main__":
    unittest.main()
