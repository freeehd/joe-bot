import unittest
from datetime import datetime, timedelta, timezone

from database.db import AuditStore
from execution.broker import AccountSnapshot
from execution.exit_engine import ExitEngine
from live.paper_engine import PaperTradingEngine
from live.runtime import PaperRuntime, RuntimeConfig
from market.stream import QuoteEvent, StreamHealth
from risk.risk_engine import ProductionRiskEngine, RiskLimits


UTC = timezone.utc


class Broker:
    paper = True
    def get_account(self): return AccountSnapshot(10_000, 10_000, 10_000)
    def get_positions(self): return []
    def get_open_orders(self): return []
    def submit_market_order(self, intent): raise AssertionError("no entries expected")
    def cancel_order(self, broker_order_id): pass


class FakeMarketStream:
    def __init__(self, now):
        self.health = StreamHealth()
        self.health.mark_connected(now)
        self.health.mark_event(QuoteEvent("AAPL", now, 99.9, 100.1, received_at=now))
    def run(self): pass
    def stop(self): self.health.mark_disconnected()


class FakeTradeStream:
    def __init__(self):
        self.connected = True
        self.last_error = None
    def run(self): pass
    def stop(self): self.connected = False


class LiveRuntimeTests(unittest.TestCase):
    def test_watchdog_sets_market_stale_kill_switch(self):
        start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        now = [start]
        broker = Broker()
        engine = PaperTradingEngine(
            broker=broker,
            risk_engine=ProductionRiskEngine(RiskLimits(stale_market_data_seconds=5)),
            exit_engine=ExitEngine(),
            audit_store=AuditStore(":memory:"),
        )
        market = FakeMarketStream(start)
        trade = FakeTradeStream()
        runtime = PaperRuntime(
            engine, market, trade,
            config=RuntimeConfig(reconciliation_interval_seconds=60),
            clock=lambda: now[0],
            sleeper=lambda _: None,
        )
        engine.start(now=start)
        runtime._last_reconciliation = start
        runtime.watchdog_once()
        self.assertNotIn("websocket_unstable", engine.state(start).kill_switches)
        now[0] = start + timedelta(seconds=6)
        runtime.watchdog_once()
        state = engine.state(now[0])
        self.assertIn("websocket_unstable", state.kill_switches)
        self.assertIn("market_data_stale", state.kill_switches)


if __name__ == "__main__":
    unittest.main()
