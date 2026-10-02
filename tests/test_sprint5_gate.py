import unittest

from research.sprint5_gate import build_sprint5_gate
from strategy.ood import OODResult
from strategy.safety_gate import MetaSafetyGate
from strategy.uncertainty import UncertaintyState


class Sprint5GateTests(unittest.TestCase):
    def test_safety_gate_can_reduce_or_veto_but_not_expand_risk(self):
        uncertainty = UncertaintyState(.3, .1, .4, .7, False)
        decision = MetaSafetyGate().evaluate(uncertainty, OODResult(1.1, 1.0, True, .5))
        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.risk_multiplier, .35)
        veto = MetaSafetyGate().evaluate(uncertainty, OODResult(3.0, 1.0, True, 0.0))
        self.assertFalse(veto.approved)
        self.assertEqual(veto.risk_multiplier, 0.0)

    def test_evidence_gate_requires_all_additive_components(self):
        result = build_sprint5_gate(
            laya_result={"gated_portfolio": {"expectancy_delta_vs_ungated_bps": 1.2, "worst_drawdown_delta_vs_ungated": -.001}},
            uncertainty_report={"sufficient_data": True, "high_uncertainty_accuracy_lower": True, "accuracy_delta_high_minus_low": -.12},
            ood_report={"sufficient_data": True, "ood_is_harder": True, "ood_accuracy_delta": -.18},
        )
        self.assertTrue(result["passed"])
        failed = build_sprint5_gate(
            laya_result={"gated_portfolio": {"expectancy_delta_vs_ungated_bps": -1.0, "worst_drawdown_delta_vs_ungated": 0.0}},
            uncertainty_report={"sufficient_data": True, "high_uncertainty_accuracy_lower": True},
            ood_report={"sufficient_data": True, "ood_is_harder": True},
        )
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
