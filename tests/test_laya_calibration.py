import unittest

import numpy as np

from models.laya_calibration import expected_calibration_error, fit_temperature, temperature_scale


class LayaCalibrationTests(unittest.TestCase):
    def test_temperature_scaling_preserves_probability_simplex(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
        scaled = temperature_scale(probs, 1.7)
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
        self.assertTrue((scaled > 0).all())

    def test_temperature_fit_can_reduce_overconfidence_ece(self):
        probs = np.array([
            [0.95, 0.03, 0.02],
            [0.95, 0.03, 0.02],
            [0.95, 0.03, 0.02],
            [0.95, 0.03, 0.02],
            [0.95, 0.03, 0.02],
            [0.95, 0.03, 0.02],
        ])
        labels = np.array([0, 0, 0, 1, 1, 2])
        before = expected_calibration_error(probs, labels, bins=3)
        temperature = fit_temperature(probs, labels)
        after = expected_calibration_error(temperature_scale(probs, temperature), labels, bins=3)
        self.assertGreater(temperature, 1.0)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
