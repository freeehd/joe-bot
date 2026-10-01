import unittest
from unittest.mock import MagicMock
import sys

# Scanner imports Alpaca at module load. Stub only the broker-facing module so
# this pure decision helper can be tested offline without network credentials.
fake_alpaca = MagicMock()
sys.modules.setdefault("market.alpaca_client", fake_alpaca)

from market.scanner import select_legacy_direction


class LegacyDirectionTests(unittest.TestCase):
    def test_unclear_when_gap_is_small(self):
        direction, probability, gap = select_legacy_direction(0.40, 0.37)
        self.assertEqual(direction, "UNCLEAR")
        self.assertAlmostEqual(probability, 0.40)
        self.assertAlmostEqual(gap, 0.03)

    def test_long_when_long_probability_dominates(self):
        direction, probability, _ = select_legacy_direction(0.70, 0.20)
        self.assertEqual(direction, "LONG")
        self.assertAlmostEqual(probability, 0.70)

    def test_short_when_short_probability_dominates(self):
        direction, probability, _ = select_legacy_direction(0.20, 0.70)
        self.assertEqual(direction, "SHORT")
        self.assertAlmostEqual(probability, 0.70)


if __name__ == "__main__":
    unittest.main()
