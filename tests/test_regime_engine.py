import unittest

from strategy.regime import REGIMES, RegimeEngine


class RegimeEngineTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_trend_direction_changes(self):
        engine = RegimeEngine()
        up = engine.infer_row({
            "spy_return_5m": 0.008, "spy_return_15m": 0.012,
            "qqq_return_5m": 0.010, "qqq_return_15m": 0.014,
            "breadth_up_1m": 0.85, "realized_vol_5m": 0.002,
            "realized_vol_15m": 0.003, "range_expansion_20": 1.0,
            "tod_relative_volume": 1.1,
        })
        down = engine.infer_row({
            "spy_return_5m": -0.008, "spy_return_15m": -0.012,
            "qqq_return_5m": -0.010, "qqq_return_15m": -0.014,
            "breadth_up_1m": 0.15, "realized_vol_5m": 0.002,
            "realized_vol_15m": 0.003, "range_expansion_20": 1.0,
            "tod_relative_volume": 1.1,
        })
        self.assertAlmostEqual(sum(up.probabilities.values()), 1.0)
        self.assertEqual(set(up.probabilities), set(REGIMES))
        self.assertGreater(up.probabilities["TREND_UP"], up.probabilities["TREND_DOWN"])
        self.assertGreater(down.probabilities["TREND_DOWN"], down.probabilities["TREND_UP"])

    def test_shock_probability_rises_with_market_move_and_volatility(self):
        engine = RegimeEngine()
        calm = engine.infer_row({"breadth_up_1m": 0.5, "range_expansion_20": 0.8, "tod_relative_volume": 0.8})
        shock = engine.infer_row({
            "spy_return_5m": -0.03, "qqq_return_5m": -0.035,
            "realized_vol_5m": 0.02, "realized_vol_15m": 0.01,
            "range_expansion_20": 3.0, "tod_relative_volume": 3.0,
            "breadth_up_1m": 0.05,
        })
        self.assertGreater(shock.probabilities["SHOCK"], calm.probabilities["SHOCK"])


if __name__ == "__main__":
    unittest.main()
