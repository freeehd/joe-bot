import unittest

from strategy.ensemble import RegimeAwareEnsemble
from strategy.regime import RegimeState
from strategy.specialists import MeanReversionSpecialist, MomentumSpecialist, default_specialists


class SpecialistEnsembleTests(unittest.TestCase):
    def test_specialists_emit_valid_shared_contract(self):
        regime = RegimeState("TREND_UP", 0.8, {
            "TREND_UP": .8, "TREND_DOWN": .02, "RANGE": .05, "HIGH_VOL": .04,
            "LOW_VOL": .03, "SHOCK": .01, "OPENING_VOLATILITY": .05,
        })
        row = {
            "return_5m": .01, "return_15m": .015, "ema_5_20_separation": .006,
            "trend_persistence_10": .8, "relative_strength_spy_5m": .004,
            "tod_relative_volume": 1.4,
        }
        signal = MomentumSpecialist().evaluate(row, regime)
        self.assertAlmostEqual(signal.p_wait + signal.p_long + signal.p_short, 1.0)
        self.assertGreaterEqual(signal.uncertainty, 0)
        self.assertLessEqual(signal.uncertainty, 1)
        self.assertEqual(signal.direction, "LONG")

    def test_regime_weighting_changes_strategy_authority(self):
        row = {"vwap_distance": .01, "price_to_ema_20": .008, "return_5m": .004, "trend_persistence_10": .05}
        range_regime = RegimeState("RANGE", .8, {
            "TREND_UP": .02, "TREND_DOWN": .02, "RANGE": .8, "HIGH_VOL": .03,
            "LOW_VOL": .08, "SHOCK": .01, "OPENING_VOLATILITY": .04,
        })
        signals = [s.evaluate(row, range_regime) for s in default_specialists()]
        combined = RegimeAwareEnsemble().combine(signals, range_regime)
        self.assertGreater(combined.specialist_weights["mean_reversion"], combined.specialist_weights["momentum"])
        self.assertAlmostEqual(combined.p_wait + combined.p_long + combined.p_short, 1.0)

    def test_disagreement_increases_wait_pressure(self):
        trend = RegimeState("RANGE", .7, {
            "TREND_UP": .05, "TREND_DOWN": .05, "RANGE": .7, "HIGH_VOL": .05,
            "LOW_VOL": .1, "SHOCK": .01, "OPENING_VOLATILITY": .04,
        })
        row = {"tod_relative_volume": 1.0}
        a = MomentumSpecialist()._finish(3.0, row, trend)
        b = MeanReversionSpecialist()._finish(-3.0, row, trend)
        combined = RegimeAwareEnsemble(disagreement_wait_boost=.8).combine([a, b], trend)
        self.assertGreater(combined.p_wait, min(a.p_wait, b.p_wait))


if __name__ == "__main__":
    unittest.main()
