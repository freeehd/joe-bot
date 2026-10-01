import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from backtest.portfolio import PortfolioBacktestConfig, run_portfolio_backtest
from strategy.ev_model import EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2, PortfolioConstraints


class PortfolioBacktestTests(unittest.TestCase):
    def test_positive_ev_candidates_are_sized_and_realized_at_account_level(self):
        index = pd.date_range("2026-01-05 14:30Z", periods=4, freq="min")
        raw = {
            "AAA": pd.DataFrame(
                [
                    [100.0, 100.1, 99.9, 100.0],
                    [100.0, 101.2, 99.9, 101.0],
                    [101.0, 101.1, 100.9, 101.0],
                    [101.0, 101.1, 100.9, 101.0],
                ],
                columns=["open", "high", "low", "close"],
                index=index,
            ),
            "BBB": pd.DataFrame(
                [
                    [50.0, 50.1, 49.9, 50.0],
                    [50.0, 50.7, 49.9, 50.5],
                    [50.5, 50.6, 50.4, 50.5],
                    [50.5, 50.6, 50.4, 50.5],
                ],
                columns=["open", "high", "low", "close"],
                index=index,
            ),
        }
        signals = pd.DataFrame(
            {
                "training_symbol": ["AAA", "BBB"],
                "p_wait": [0.15, 0.20],
                "p_long": [0.80, 0.75],
                "p_short": [0.05, 0.05],
                "direction": ["LONG", "LONG"],
                "confidence": [0.80, 0.75],
            },
            index=[index[0], index[0]],
        )
        trade = TradeConfig(target_pct=0.01, stop_pct=0.01, max_holding_bars=2)
        execution = ExecutionConfig(spread_bps=0, slippage_bps=0)
        ev = EmpiricalEVModel(trade_config=trade, execution_config=execution)
        allocator = PortfolioAllocatorV2(
            trade_config=trade,
            constraints=PortfolioConstraints(
                max_positions=2,
                max_sector_deployed_fraction=0.80,
                max_correlated_risk_fraction=0.01,
            ),
        )
        result = run_portfolio_backtest(
            signals,
            raw_loader=lambda symbol: raw[symbol],
            ranker=EVRanker(ev),
            allocator=allocator,
            trade_config=trade,
            execution_config=execution,
            config=PortfolioBacktestConfig(initial_equity=10_000),
        )
        self.assertEqual(result["trade_metrics"]["total_trades"], 2)
        self.assertGreater(result["ending_equity"], 10_000)
        self.assertGreater(result["total_return"], 0)
        self.assertEqual(len(result["trade_records"]), 2)
        self.assertEqual(len(result["allocation_snapshots"]), 1)


if __name__ == "__main__":
    unittest.main()
