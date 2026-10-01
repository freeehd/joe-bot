import unittest

from market.universe import (
    LEGACY_SYMBOLS,
    MARKET_CONTEXT_SYMBOLS,
    RESEARCH_UNIVERSE_50,
    SYMBOLS,
)


class UniverseTests(unittest.TestCase):
    def test_phase_b_universe_has_50_unique_equities(self):
        self.assertEqual(len(RESEARCH_UNIVERSE_50), 50)
        self.assertEqual(len(set(RESEARCH_UNIVERSE_50)), 50)
        self.assertTrue(set(RESEARCH_UNIVERSE_50).isdisjoint(MARKET_CONTEXT_SYMBOLS))

    def test_legacy_symbols_are_preserved_independently(self):
        self.assertEqual(SYMBOLS, LEGACY_SYMBOLS)
        self.assertEqual(MARKET_CONTEXT_SYMBOLS, ["SPY", "QQQ"])


if __name__ == "__main__":
    unittest.main()
