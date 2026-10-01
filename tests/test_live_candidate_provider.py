import unittest
from datetime import datetime, timezone

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from execution.broker import AccountSnapshot
from live.candidate_provider import LiveCandidateProvider, LiveProviderConfig
from live.feature_store import FeatureBatch
from live.paper_engine import PaperEngineState
from market.stream import BarEvent
from strategy.ev_model import EmpiricalEVModel
from strategy.portfolio_allocator import PortfolioAllocatorV2, PortfolioConstraints

UTC = timezone.utc


class FakeFeatures:
    def __init__(self, frame):
        self.frame = frame
    def add_bar(self, event):
        return FeatureBatch(pd.Timestamp(event.timestamp), self.frame, 1.0)


class FakeAlpha:
    def predict_frame(self, frame):
        return [
            {"p_wait": 0.15, "p_long": 0.80, "p_short": 0.05, "direction": "LONG", "confidence": 0.80}
            for _ in range(len(frame))
        ]


class BrokenAlpha:
    def predict_frame(self, frame):
        raise RuntimeError("bad model")


class LiveCandidateProviderTests(unittest.TestCase):
    def setUp(self):
        ts = pd.Timestamp("2026-01-05T14:30:00Z")
        self.frame = pd.DataFrame([
            {"training_symbol": "AAPL", "close": 100.0, "realized_vol_5m": 0.001},
            {"training_symbol": "SPY", "close": 500.0, "realized_vol_5m": 0.001},
            {"training_symbol": "QQQ", "close": 450.0, "realized_vol_5m": 0.001},
        ], index=[ts, ts, ts])
        self.event = BarEvent("QQQ", ts.to_pydatetime(), 450, 451, 449, 450, 1000, received_at=ts.to_pydatetime())
        self.state = PaperEngineState(AccountSnapshot(10_000, 10_000, 10_000), (), True, ())
        self.ev = EmpiricalEVModel(
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.0015, max_holding_bars=10),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        ).fit(pd.DataFrame(columns=["side", "confidence", "net_return"]))

    def test_full_stack_emits_risk_sized_entry_proposal(self):
        provider = LiveCandidateProvider(
            FakeFeatures(self.frame), FakeAlpha(), self.ev,
            allocator=PortfolioAllocatorV2(
                constraints=PortfolioConstraints(max_positions=3, min_position_dollars=1),
                trade_config=self.ev.trade_config,
            ),
            config=LiveProviderConfig(minimum_direction_confidence=0.6),
        )
        proposals = provider(self.event, self.state)
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.symbol, "AAPL")
        self.assertEqual(proposal.direction, "LONG")
        self.assertGreater(proposal.expected_ev_bps, 0)
        self.assertGreater(proposal.quantity, 0)
        self.assertLess(proposal.stop_price, proposal.reference_price)
        self.assertGreater(proposal.target_price, proposal.reference_price)

    def test_model_failure_fails_closed(self):
        provider = LiveCandidateProvider(FakeFeatures(self.frame), BrokenAlpha(), self.ev)
        self.assertEqual(provider(self.event, self.state), [])
        self.assertTrue(provider.last_diagnostics["failed_closed"])


if __name__ == "__main__":
    unittest.main()
