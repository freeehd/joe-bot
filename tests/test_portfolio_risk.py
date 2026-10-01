import unittest

import pandas as pd

from risk.portfolio_risk import candidate_correlation_penalty, directional_correlation


class PortfolioRiskTests(unittest.TestCase):
    def test_directional_correlation_recognizes_hedge(self):
        self.assertAlmostEqual(directional_correlation(0.9, "LONG", "LONG"), 0.9)
        self.assertAlmostEqual(directional_correlation(0.9, "LONG", "SHORT"), -0.9)
        self.assertAlmostEqual(directional_correlation(-0.8, "LONG", "SHORT"), 0.8)

    def test_correlation_penalty_only_penalizes_concentrated_pnl_exposure(self):
        corr = pd.DataFrame(
            [[1.0, 0.9], [0.9, 1.0]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        )
        same_multiplier, same_corr = candidate_correlation_penalty(
            symbol="AAA",
            side="LONG",
            positions=[{"symbol": "BBB", "direction": "LONG"}],
            correlations=corr,
        )
        hedge_multiplier, hedge_corr = candidate_correlation_penalty(
            symbol="AAA",
            side="LONG",
            positions=[{"symbol": "BBB", "direction": "SHORT"}],
            correlations=corr,
        )
        self.assertLess(same_multiplier, 1.0)
        self.assertAlmostEqual(same_corr, 0.9)
        self.assertEqual(hedge_multiplier, 1.0)
        self.assertEqual(hedge_corr, 0.0)


if __name__ == "__main__":
    unittest.main()
