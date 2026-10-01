import unittest

import pandas as pd

from models.train_multiclass import chronological_purged_split


class ChronologicalSplitTests(unittest.TestCase):
    def test_split_is_ordered_and_purged(self):
        timestamps = pd.date_range("2026-01-02 09:30", periods=100, freq="min")
        rows = []
        for symbol in ["AAA", "BBB"]:
            frame = pd.DataFrame(
                {
                    "training_symbol": symbol,
                    "trade_label": 0,
                },
                index=timestamps,
            )
            rows.append(frame)
        df = pd.concat(rows).sort_index()

        train, calibration, test = chronological_purged_split(
            df,
            train_fraction=0.70,
            calibration_fraction=0.15,
            purge_bars=5,
        )

        self.assertLess(train.index.max(), calibration.index.min())
        self.assertLess(calibration.index.max(), test.index.min())

        unique = pd.Index(df.index.unique()).sort_values()
        train_boundary = int(len(unique) * 0.70)
        calibration_boundary = int(len(unique) * 0.85)
        self.assertEqual(train.index.max(), unique[train_boundary - 6])
        self.assertEqual(calibration.index.min(), unique[train_boundary])
        self.assertEqual(calibration.index.max(), unique[calibration_boundary - 6])
        self.assertEqual(test.index.min(), unique[calibration_boundary])


if __name__ == "__main__":
    unittest.main()
