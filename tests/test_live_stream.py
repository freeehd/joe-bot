import unittest
from datetime import datetime, timedelta, timezone

from market.stream import BarEvent, QuoteEvent, StreamHealth


UTC = timezone.utc


class LiveStreamTests(unittest.TestCase):
    def test_quote_spread_and_stream_staleness(self):
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        quote = QuoteEvent("aapl", now, 99.95, 100.05, received_at=now)
        self.assertEqual(quote.symbol, "AAPL")
        self.assertAlmostEqual(quote.mid, 100.0)
        self.assertAlmostEqual(quote.spread_bps, 10.0)

        health = StreamHealth()
        health.mark_connected(now)
        health.mark_event(quote)
        self.assertFalse(health.is_stale(5, now=now + timedelta(seconds=4)))
        self.assertTrue(health.is_stale(5, now=now + timedelta(seconds=6)))

    def test_invalid_ohlc_is_rejected(self):
        now = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        with self.assertRaises(ValueError):
            BarEvent("AAPL", now, 100, 99, 98, 100, 1000)


if __name__ == "__main__":
    unittest.main()
