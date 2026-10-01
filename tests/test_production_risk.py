import unittest
from datetime import datetime, timedelta, timezone

from risk.risk_engine import KillSwitchReason, ProductionRiskEngine, RiskLimits


UTC = timezone.utc


class ProductionRiskTests(unittest.TestCase):
    def _healthy(self, engine, now):
        engine.update_health(
            broker_connected=True,
            websocket_stable=True,
            model_ready=True,
            features_valid=True,
            position_state_consistent=True,
            unexpected_volatility=False,
            broker_trading_blocked=False,
            clock_drift_seconds=0.0,
        )
        engine.mark_market_event(now)

    def test_stale_market_data_blocks_entries_but_not_exits(self):
        now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
        engine = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=5))
        engine.reset_session(10_000)
        self._healthy(engine, now - timedelta(seconds=6))
        decision = engine.approve_entry(
            account_equity=10_000, proposed_notional=1000, open_positions=0,
            gross_exposure=0, net_exposure=0, sector_exposure=0, direction="LONG", now=now,
        )
        self.assertFalse(decision.approved)
        self.assertIn(KillSwitchReason.MARKET_DATA_STALE.value, decision.kill_switches)
        self.assertTrue(engine.approve_exit().approved)

    def test_daily_loss_and_position_limits_are_independent(self):
        now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
        engine = ProductionRiskEngine(RiskLimits(max_daily_loss_fraction=0.01, stale_market_data_seconds=60))
        engine.reset_session(10_000)
        self._healthy(engine, now)
        engine.update_equity(9_890)
        self.assertIn(KillSwitchReason.DAILY_LOSS, engine.active_kill_switches(now))


if __name__ == "__main__":
    unittest.main()
