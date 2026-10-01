import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from strategy.ev_model import EVConfig, EmpiricalEVModel, estimated_round_trip_cost, structural_expected_value


class ExpectedValueModelTests(unittest.TestCase):
    def test_structural_ev_subtracts_round_trip_costs(self):
        trade = TradeConfig(target_pct=0.003, stop_pct=0.0015, max_holding_bars=10)
        execution = ExecutionConfig(spread_bps=4, slippage_bps=2, fee_bps=0)
        gross, net = structural_expected_value(
            side="LONG",
            p_wait=0.2,
            p_long=0.7,
            p_short=0.1,
            trade_config=trade,
            execution_config=execution,
        )
        self.assertAlmostEqual(gross, 0.00195)
        self.assertAlmostEqual(estimated_round_trip_cost(execution), 0.0008)
        self.assertAlmostEqual(net, 0.00115)

    def test_empirical_ev_is_shrunk_toward_structural_prior(self):
        model = EmpiricalEVModel(
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.0015),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
            config=EVConfig(min_samples_per_bucket=5, shrinkage_samples=20),
        )
        trades = pd.DataFrame(
            {
                "side": ["LONG"] * 20,
                "confidence": [0.72] * 20,
                "net_return": [0.002] * 20,
            }
        )
        model.fit(trades)
        estimate = model.estimate(
            side="LONG",
            p_wait=0.1,
            p_long=0.72,
            p_short=0.18,
            confidence=0.72,
        )
        self.assertEqual(estimate.empirical_samples, 20)
        self.assertAlmostEqual(estimate.empirical_weight, 0.5)
        self.assertIsNotNone(estimate.empirical_net_ev)
        self.assertGreater(estimate.blended_net_ev, estimate.structural_net_ev)
        self.assertLess(estimate.blended_net_ev, 0.002)

    def test_sparse_empirical_bucket_does_not_override_prior(self):
        model = EmpiricalEVModel(
            trade_config=TradeConfig(),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
            config=EVConfig(min_samples_per_bucket=10, shrinkage_samples=20),
        ).fit(
            pd.DataFrame(
                {
                    "side": ["SHORT"] * 3,
                    "confidence": [0.75] * 3,
                    "net_return": [0.05] * 3,
                }
            )
        )
        estimate = model.estimate(
            side="SHORT",
            p_wait=0.1,
            p_long=0.15,
            p_short=0.75,
            confidence=0.75,
        )
        self.assertEqual(estimate.empirical_samples, 3)
        self.assertEqual(estimate.empirical_weight, 0.0)
        self.assertAlmostEqual(estimate.blended_net_ev, estimate.structural_net_ev)


if __name__ == "__main__":
    unittest.main()
