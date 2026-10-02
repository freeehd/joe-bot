import unittest

import numpy as np
import pandas as pd

from strategy.ood import RobustOODDetector
from strategy.uncertainty import disagreement_state


class UncertaintyOODTests(unittest.TestCase):
    def test_disagreement_reduces_risk_and_can_force_wait(self):
        agreeing = disagreement_state([[.05, .90, .05], [.08, .86, .06]], force_wait_threshold=.9)
        disagreeing = disagreement_state([[.05, .90, .05], [.05, .05, .90]], force_wait_threshold=.65)
        self.assertGreater(agreeing.risk_multiplier, disagreeing.risk_multiplier)
        self.assertTrue(disagreeing.force_wait)
        self.assertEqual(disagreeing.risk_multiplier, 0.0)

    def test_ood_threshold_is_calibrated_from_earlier_partition(self):
        rng = np.random.default_rng(7)
        train = pd.DataFrame(rng.normal(0, 1, size=(200, 3)), columns=["a", "b", "c"])
        calibration = pd.DataFrame(rng.normal(0, 1, size=(80, 3)), columns=["a", "b", "c"])
        detector = RobustOODDetector(quantile=.95).fit(train, calibration, ["a", "b", "c"])
        normal = detector.evaluate(pd.Series({"a": 0.1, "b": -.1, "c": .2}))
        extreme = detector.evaluate(pd.Series({"a": 25.0, "b": 25.0, "c": 25.0}))
        self.assertFalse(normal.is_ood)
        self.assertTrue(extreme.is_ood)
        self.assertEqual(extreme.risk_multiplier, 0.0)


if __name__ == "__main__":
    unittest.main()
