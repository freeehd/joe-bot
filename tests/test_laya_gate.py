import unittest

from models.laya_engine import LayaDecision
from strategy.laya_gate import LayaGateConfig, LayaVetoGate


class _FakeEngine:
    def __init__(self, decisions):
        self.decisions = decisions
        self.seen_states = []

    def evaluate_states(self, states):
        self.seen_states.extend(states)
        return self.decisions[: len(states)]


def decision(action, *, quality="GOOD", risk=0.1, confidence=0.8):
    action_probs = {"LONG": 0.1, "WAIT": 0.1, "SHORT": 0.1}
    action_probs[action] = confidence
    return LayaDecision(
        action=action,
        action_probabilities=action_probs,
        action_answer_confidence=confidence,
        quality=quality,
        quality_probabilities={"POOR": 0.1, "FAIR": 0.1, "GOOD": 0.7, "EXCELLENT": 0.1},
        quality_answer_confidence=0.7,
        risk_concern_probability=risk,
        model="fake",
    )


class LayaGateTests(unittest.TestCase):
    def test_gate_can_only_veto_not_create_or_flip_trades(self):
        candidates = [
            {"symbol": "AAA", "direction": "LONG", "p_wait": 0.1, "p_long": 0.8, "p_short": 0.1, "confidence": 0.8, "net_ev_bps": 8.0},
            {"symbol": "BBB", "direction": "SHORT", "p_wait": 0.1, "p_long": 0.1, "p_short": 0.8, "confidence": 0.8, "net_ev_bps": 7.0},
        ]
        gate = LayaVetoGate(_FakeEngine([decision("LONG"), decision("LONG")]))
        result = gate.gate_candidates(candidates)
        self.assertEqual([(x["symbol"], x["direction"]) for x in result["approved"]], [("AAA", "LONG")])
        self.assertEqual(result["vetoed"][0]["symbol"], "BBB")

    def test_risk_and_poor_quality_are_vetoes(self):
        candidates = [
            {"symbol": "AAA", "direction": "LONG", "p_wait": 0.1, "p_long": 0.8, "p_short": 0.1, "confidence": 0.8, "net_ev_bps": 8.0},
            {"symbol": "BBB", "direction": "LONG", "p_wait": 0.1, "p_long": 0.8, "p_short": 0.1, "confidence": 0.8, "net_ev_bps": 7.0},
        ]
        gate = LayaVetoGate(
            _FakeEngine([decision("LONG", risk=0.9), decision("LONG", quality="POOR")]),
            config=LayaGateConfig(risk_veto_probability=0.7),
        )
        result = gate.gate_candidates(candidates)
        self.assertEqual(result["approved"], [])
        self.assertEqual(len(result["vetoed"]), 2)

    def test_only_top_quant_candidates_reach_laya(self):
        candidates = [
            {"symbol": symbol, "direction": "LONG", "p_wait": 0.1, "p_long": 0.8, "p_short": 0.1, "confidence": 0.8, "net_ev_bps": 10-i}
            for i, symbol in enumerate(["AAA", "BBB", "CCC"])
        ]
        engine = _FakeEngine([decision("LONG"), decision("LONG")])
        gate = LayaVetoGate(engine, config=LayaGateConfig(top_candidates=2))
        result = gate.gate_candidates(candidates)
        self.assertEqual(len(engine.seen_states), 2)
        self.assertEqual([x["symbol"] for x in result["approved"]], ["AAA", "BBB"])


if __name__ == "__main__":
    unittest.main()
