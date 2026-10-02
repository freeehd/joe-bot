import unittest

from research.core_edge import CoreEdgeThresholds, build_core_edge_report, ev_monotonicity


class CoreEdgeTests(unittest.TestCase):
    def test_ev_monotonicity_detects_ordered_realized_returns(self):
        import pandas as pd
        frame = pd.DataFrame({
            "net_ev_bps_at_entry": list(range(10, 110, 10)),
            "net_return": [x / 10000 for x in range(2, 22, 2)],
            "symbol": ["A", "B"] * 5,
        })
        result = ev_monotonicity(frame, buckets=5)
        self.assertTrue(result["sufficient_data"])
        self.assertGreater(result["spearman"], 0.9)
        self.assertEqual(result["adjacent_monotonic_rate"], 1.0)

    def test_core_edge_report_can_explicitly_pass(self):
        records = []
        for i in range(40):
            ev = 5 + i
            records.append({"symbol": f"S{i % 8}", "net_ev_bps_at_entry": ev, "net_return": (ev * 0.5) / 10000})
        classification = {"average_metrics": {"ece_10": 0.04}}
        v06 = {
            "combined_trade_metrics": {"expectancy_bps": 4.0},
            "positive_expectancy_window_rate": 0.8,
            "execution_stress": {
                "double_spread": {"expectancy_bps": 2.0},
                "combined_adverse": {"expectancy_bps": 1.0},
            },
        }
        v07 = {
            "portfolio_trade_metrics": {"expectancy_bps": 6.0},
            "baseline_trade_metrics": {"expectancy_bps": 4.0},
            "expectancy_improvement_bps": 2.0,
            "portfolio_positive_return_window_rate": 0.8,
            "portfolio_worst_window_drawdown": -0.05,
            "portfolio_trade_records": records,
        }
        report = build_core_edge_report(
            classification_walk_forward=classification,
            v06_walk_forward=v06,
            v07_portfolio_walk_forward=v07,
            thresholds=CoreEdgeThresholds(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["verdict"], "PASS")

    def test_core_edge_report_fails_negative_stress(self):
        classification = {"average_metrics": {"ece_10": 0.04}}
        v06 = {"combined_trade_metrics": {"expectancy_bps": 4.0}, "execution_stress": {"combined_adverse": {"expectancy_bps": -1.0}}}
        v07 = {
            "portfolio_trade_metrics": {"expectancy_bps": 5.0},
            "portfolio_positive_return_window_rate": 0.8,
            "portfolio_worst_window_drawdown": -0.04,
            "portfolio_trade_records": [
                {"symbol": f"S{i%10}", "net_ev_bps_at_entry": i + 1, "net_return": (i + 1) / 100000}
                for i in range(40)
            ],
        }
        report = build_core_edge_report(classification_walk_forward=classification, v06_walk_forward=v06, v07_portfolio_walk_forward=v07)
        self.assertFalse(report["passed"])
        failed = {g["name"] for g in report["gates"] if not g["passed"]}
        self.assertIn("positive_under_all_configured_cost_stress", failed)


if __name__ == "__main__":
    unittest.main()
