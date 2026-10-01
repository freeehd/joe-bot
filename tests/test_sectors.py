import unittest

from market.sectors import sector_for
from market.universe import MARKET_CONTEXT_SYMBOLS, RESEARCH_UNIVERSE_50


class SectorMetadataTests(unittest.TestCase):
    def test_research_universe_has_explicit_sector_metadata(self):
        for symbol in RESEARCH_UNIVERSE_50 + MARKET_CONTEXT_SYMBOLS:
            self.assertNotEqual(sector_for(symbol), "UNKNOWN", symbol)

    def test_unknown_symbol_is_conservative_unknown(self):
        self.assertEqual(sector_for("NOT_A_REAL_SYMBOL"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
