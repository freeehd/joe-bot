import unittest

import numpy as np
import pandas as pd

from backtest.signals import probability_frame


class SignalFrameTests(unittest.TestCase):
    def test_probabilities_become_direction_and_confidence(self):
        frame = pd.DataFrame(
            {"training_symbol": ["AAA", "BBB"]},
            index=pd.to_datetime(["2026-01-05 14:30Z", "2026-01-05 14:31Z"]),
        )
        probabilities = np.array([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]])
        result = probability_frame(frame, probabilities)
        self.assertEqual(result.iloc[0]["direction"], "LONG")
        self.assertEqual(result.iloc[1]["direction"], "WAIT")
        self.assertAlmostEqual(result.iloc[0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
