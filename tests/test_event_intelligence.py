import unittest
from datetime import datetime, timedelta, timezone

from events.intelligence import validate_event_payload, aggregate_symbol_events

UTC=timezone.utc


class EventIntelligenceTests(unittest.TestCase):
    def test_validated_event_becomes_feature_not_trade(self):
        now=datetime(2026,1,5,15,0,tzinfo=UTC)
        event=validate_event_payload({
            "event_type":"earnings_guidance","direction":"positive","surprise":.8,
            "impact":"high","expected_horizon":"intraday","credibility":.95,
            "symbols":["aapl"],"published_at":now.isoformat(),"source_id":"wire-1",
        })
        self.assertEqual(event.symbols,("AAPL",))
        self.assertGreater(event.to_features()["event_score"],0)
        self.assertFalse(hasattr(event,"submit_order"))

    def test_aggregation_decays_old_events(self):
        now=datetime(2026,1,5,15,0,tzinfo=UTC)
        fresh=validate_event_payload({"event_type":"x","direction":"positive","surprise":1,"impact":"high","expected_horizon":"intraday","credibility":1,"symbols":["AAPL"],"published_at":now,"source_id":"1"})
        old=validate_event_payload({"event_type":"x","direction":"negative","surprise":1,"impact":"high","expected_horizon":"intraday","credibility":1,"symbols":["AAPL"],"published_at":now-timedelta(minutes=500),"source_id":"2"})
        agg=aggregate_symbol_events([fresh,old],now=now,max_age_minutes=390)
        self.assertEqual(agg["AAPL"]["event_count"],1.0)
        self.assertGreater(agg["AAPL"]["event_score"],0)


if __name__ == "__main__":
    unittest.main()
