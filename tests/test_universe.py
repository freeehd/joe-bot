import unittest

from market.universe import LEGACY_SYMBOLS, RESEARCH_UNIVERSE_50, SYMBOLS


class UniverseTests(unittest.TestCase):
    def test_phase_b_universe_has_50_unique_symbols(self):
        self.assertEqual(len(RESEARCH_UNIVERSE_50), 50)
        self.assertEqual(len(set(RESEARCH_UNIVERSE_50)), 50)

    def test_legacy_symbols_are_preserved(self):
        self.assertEqual(SYMBOLS, LEGACY_SYMBOLS)
        self.assertTrue(set(LEGACY_SYMBOLS).issubset(RESEARCH_UNIVERSE_50))


if __name__ == "__main__":
    unittest.main()
