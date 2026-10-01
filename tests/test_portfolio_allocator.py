import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from strategy.ev_model import EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2, PortfolioConstraints


class PortfolioAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.trade = TradeConfig(target_pct=0.01, stop_pct=0.01, max_holding_bars=5)
        self.ev = EmpiricalEVModel(
            trade_config=self.trade,
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        )

    def _candidate(self, symbol, direction="LONG", confidence=0.8, price=100.0, sector="TECH"):
        if direction == "LONG":
            p_long, p_short = confidence, 0.05
        else:
            p_long, p_short = 0.05, confidence
        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "p_wait": 1.0 - p_long - p_short,
            "p_long": p_long,
            "p_short": p_short,
            "price": price,
            "sector": sector,
        }

    def test_no_positive_ev_keeps_capital_idle(self):
        ranker = EVRanker(self.ev)
        weak = {
            "symbol": "AAA",
            "direction": "LONG",
            "confidence": 0.34,
            "p_wait": 0.34,
            "p_long": 0.33,
            "p_short": 0.33,
            "price": 100.0,
            "sector": "TECH",
        }
        ranked = ranker.rank([weak], min_net_ev_bps=0.0)
        result = PortfolioAllocatorV2(trade_config=self.trade).allocate(
            ranked,
            account_equity=10_000,
        )
        self.assertEqual(result["selected_count"], 0)
        self.assertEqual(result["capital_deployed"], 0.0)

    def test_position_size_is_risk_based_then_notional_capped(self):
        ranker = EVRanker(self.ev)
        ranked = ranker.rank([self._candidate("AAA")])
        allocator = PortfolioAllocatorV2(
            trade_config=self.trade,
            constraints=PortfolioConstraints(
                risk_per_trade_fraction=0.0025,
                max_position_fraction=0.30,
                max_correlated_risk_fraction=0.01,
            ),
        )
        result = allocator.allocate(ranked, account_equity=10_000)
        position = result["positions"][0]
        # $25 risk budget / $1 risk per share = 25 shares = $2,500 notional.
        self.assertEqual(position["quantity"], 25)
        self.assertAlmostEqual(position["allocation"], 2500.0)
        self.assertAlmostEqual(position["risk_dollars"], 25.0)

    def test_sector_cap_reduces_second_position(self):
        ranker = EVRanker(self.ev)
        ranked = ranker.rank([self._candidate("AAA"), self._candidate("BBB", confidence=0.75)])
        allocator = PortfolioAllocatorV2(
            trade_config=self.trade,
            constraints=PortfolioConstraints(
                max_sector_deployed_fraction=0.30,
                risk_per_trade_fraction=0.0025,
                max_correlated_risk_fraction=0.01,
            ),
        )
        result = allocator.allocate(ranked, account_equity=10_000)
        self.assertEqual(result["selected_count"], 2)
        self.assertLessEqual(sum(p["allocation"] for p in result["positions"]), 3000.0)

    def test_dynamic_reranking_prefers_diversification_after_first_pick(self):
        corr = pd.DataFrame(
            [
                [1.0, 0.95, 0.0],
                [0.95, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            index=["AAA", "BBB", "CCC"],
            columns=["AAA", "BBB", "CCC"],
        )
        ranker = EVRanker(self.ev)
        ranked = ranker.rank(
            [
                self._candidate("AAA", confidence=0.90, sector="A"),
                self._candidate("BBB", confidence=0.89, sector="B"),
                self._candidate("CCC", confidence=0.80, sector="C"),
            ],
            correlations=corr,
        )
        allocator = PortfolioAllocatorV2(
            trade_config=self.trade,
            constraints=PortfolioConstraints(
                max_positions=2,
                max_correlated_risk_fraction=0.10,
                max_sector_deployed_fraction=0.80,
            ),
        )
        result = allocator.allocate(ranked, account_equity=10_000, correlations=corr)
        symbols = [position["symbol"] for position in result["positions"]]
        self.assertEqual(symbols, ["AAA", "CCC"])

    def test_highly_correlated_second_bet_hits_cluster_risk_cap(self):
        corr = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        )
        ranker = EVRanker(self.ev)
        ranked = ranker.rank([self._candidate("AAA"), self._candidate("BBB", confidence=0.79)], correlations=corr)
        allocator = PortfolioAllocatorV2(
            trade_config=self.trade,
            constraints=PortfolioConstraints(
                risk_per_trade_fraction=0.0025,
                max_correlated_risk_fraction=0.0025,
                correlation_threshold=0.75,
                max_sector_deployed_fraction=0.80,
            ),
        )
        result = allocator.allocate(ranked, account_equity=10_000, correlations=corr)
        self.assertEqual(result["selected_count"], 1)
        reasons = [item["reason"] for item in result["rejected"]]
        self.assertIn("portfolio/correlation risk budget exhausted", reasons)


if __name__ == "__main__":
    unittest.main()
