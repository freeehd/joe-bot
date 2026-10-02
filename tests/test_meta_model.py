import unittest

import numpy as np
import pandas as pd

from strategy.meta_model import SpecialistMetaModel


class MetaModelTests(unittest.TestCase):
    def test_fit_predict_proba_contract(self):
        n = 90
        idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
        x = np.linspace(-0.02, 0.02, n)
        frame = pd.DataFrame({
            "return_5m": x,
            "return_15m": x * 1.4,
            "ema_5_20_separation": x * .4,
            "ema_20_slope_5": x * .2,
            "ema_50_slope_10": x * .15,
            "trend_persistence_10": np.tanh(x * 80),
            "relative_strength_spy_5m": x * .3,
            "vwap_distance": x * .2,
            "price_to_ema_20": x * .2,
            "breakout_strength_20_atr": x * 20,
            "bounce_from_session_low_atr": 1.0,
            "pullback_from_session_high_atr": 1.0,
            "distance_from_low_20": .005,
            "spy_return_5m": x * .4,
            "spy_return_15m": x * .6,
            "qqq_return_5m": x * .5,
            "qqq_return_15m": x * .7,
            "breadth_up_1m": np.clip(.5 + x * 15, 0, 1),
            "realized_vol_5m": .003,
            "realized_vol_15m": .004,
            "range_expansion_20": 1.0,
            "tod_relative_volume": 1.1,
            "minutes_since_open_norm": .3,
            "trade_label": np.tile([2, 0, 1], 30),
        }, index=idx)
        model = SpecialistMetaModel().fit(frame.iloc[:60], frame.iloc[60:75])
        probs = model.predict_proba(frame.iloc[75:])
        self.assertEqual(probs.shape, (15, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
