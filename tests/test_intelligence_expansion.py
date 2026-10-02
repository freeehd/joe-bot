import unittest

from research.intelligence_expansion import build_intelligence_expansion_gate


class IntelligenceExpansionGateTests(unittest.TestCase):
    def test_gate_passes_only_when_meta_improves_core(self):
        result = {
            "expectancy_improvement_bps": 2.5,
            "core_positive_window_rate": .60,
            "meta_positive_window_rate": .70,
            "combined": {
                "core": {"trade_metrics": {"expectancy_bps": 4.0}, "regimes": []},
                "meta": {"trade_metrics": {"expectancy_bps": 6.5}, "regimes": [
                    {"regime": "TREND_UP", "expectancy_bps": 5.0},
                    {"regime": "RANGE", "expectancy_bps": 2.0},
                ]},
            },
        }
        gate = build_intelligence_expansion_gate(result)
        self.assertTrue(gate["passed"])

    def test_gate_fails_if_complexity_does_not_add_ev(self):
        result = {
            "expectancy_improvement_bps": -1.0,
            "core_positive_window_rate": .70,
            "meta_positive_window_rate": .60,
            "combined": {
                "core": {"trade_metrics": {"expectancy_bps": 5.0}, "regimes": []},
                "meta": {"trade_metrics": {"expectancy_bps": 4.0}, "regimes": [
                    {"regime": "TREND_UP", "expectancy_bps": 2.0},
                ]},
            },
        }
        gate = build_intelligence_expansion_gate(result)
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
