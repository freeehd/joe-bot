import unittest

from models.laya_engine import LayaEngine, normalize_laya_result


class _FakeAgent:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _result():
        return {
            "model": "fake-laya",
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "LONG",
                    "probabilities": {"LONG": 0.72, "WAIT": 0.20, "SHORT": 0.08},
                    "confidence": 0.31,
                    "answer_confidence": 0.72,
                },
                "quality": {
                    "type": "choice",
                    "choice": "GOOD",
                    "probabilities": {"POOR": 0.05, "FAIR": 0.15, "GOOD": 0.65, "EXCELLENT": 0.15},
                    "answer_confidence": 0.65,
                },
                "risk_concern": {"type": "noul", "noul": 0.18, "answer_confidence": 0.82},
            },
        }

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._result()

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, questions, kwargs))
        return [self._result() for _ in states]


class LayaEngineTests(unittest.TestCase):
    def test_normalization_uses_answer_confidence_not_entropy_confidence(self):
        decision = normalize_laya_result(_FakeAgent._result())
        self.assertEqual(decision.action, "LONG")
        self.assertAlmostEqual(decision.action_answer_confidence, 0.72)
        self.assertEqual(decision.quality, "GOOD")
        self.assertAlmostEqual(decision.risk_concern_probability, 0.18)

    def test_injected_agent_allows_offline_batch_testing(self):
        agent = _FakeAgent()
        engine = LayaEngine(agent=agent)
        decisions = engine.evaluate_states([{"symbol": "AAA"}, {"symbol": "BBB"}], batch_size=2)
        self.assertEqual([item.action for item in decisions], ["LONG", "LONG"])
        self.assertEqual(len(agent.calls), 1)


if __name__ == "__main__":
    unittest.main()
