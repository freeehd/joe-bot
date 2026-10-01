import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from backtest.portfolio import PortfolioBacktestConfig, run_portfolio_backtest
from strategy.ev_model import EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2, PortfolioConstraints


class _FirstOnlyGate:
    def gate_candidates(self, ranked_candidates, **context):
        approved = ranked_candidates[:1]
        vetoed = [
            {"symbol": item["symbol"], "direction": item["direction"], "reason": "test veto"}
            for item in ranked_candidates[1:]
        ]
        return {"approved": approved, "vetoed": vetoed, "decisions": []}


class PortfolioLayaGateTests(unittest.TestCase):
    def test_candidate_gate_filters_before_allocator_without_creating_trades(self):
        index = pd.date_range("2026-01-05 14:30Z", periods=3, freq="min")
        raw = {
            "AAA": pd.DataFrame(
                [[100, 100.1, 99.9, 100], [100, 101.2, 99.9, 101], [101, 101.1, 100.9, 101]],
                columns=["open", "high", "low", "close"], index=index,
            ),
            "BBB": pd.DataFrame(
                [[50, 50.1, 49.9, 50], [50, 50.7, 49.9, 50.5], [50.5, 50.6, 50.4, 50.5]],
                columns=["open", "high", "low", "close"], index=index,
            ),
        }
        signals = pd.DataFrame(
            {
                "training_symbol": ["AAA", "BBB"],
                "p_wait": [0.1, 0.15],
                "p_long": [0.8, 0.75],
                "p_short": [0.1, 0.1],
                "direction": ["LONG", "LONG"],
                "confidence": [0.8, 0.75],
                "return_1m": [0.002, 0.001],
            },
            index=[index[0], index[0]],
        )
        trade = TradeConfig(target_pct=0.01, stop_pct=0.01, max_holding_bars=2)
        execution = ExecutionConfig(spread_bps=0, slippage_bps=0)
        ev_model = EmpiricalEVModel(trade_config=trade, execution_config=execution)
        allocator = PortfolioAllocatorV2(
            trade_config=trade,
            constraints=PortfolioConstraints(
                max_positions=2,
                max_sector_deployed_fraction=1.0,
                max_long_deployed_fraction=1.0,
                max_correlated_risk_fraction=1.0,
                min_position_dollars=1.0,
            ),
        )
        result = run_portfolio_backtest(
            signals,
            raw_loader=lambda symbol: raw[symbol],
            ranker=EVRanker(ev_model),
            allocator=allocator,
            trade_config=trade,
            execution_config=execution,
            config=PortfolioBacktestConfig(initial_equity=10_000),
            candidate_gate=_FirstOnlyGate(),
        )
        self.assertEqual(result["trade_metrics"]["total_trades"], 1)
        self.assertEqual(result["gate_snapshots"][0]["laya_approved"], 1)
        self.assertEqual(result["gate_snapshots"][0]["laya_vetoed"], 1)
        self.assertEqual(result["trade_records"].iloc[0]["symbol"], "AAA")


if __name__ == "__main__":
    unittest.main()
