import unittest

from research.champion_challenger import ChallengerPolicy, ModelScorecard, evaluate_challenger


class ChampionChallengerTests(unittest.TestCase):
    def test_strong_challenger_passes_all_gates(self):
        champion = ModelScorecard("champ", 500, 6.0, 2.0, -0.06, 0.030, 0.65)
        challenger = ModelScorecard("chall", 400, 8.0, 1.0, -0.065, 0.031, 0.70)
        verdict = evaluate_challenger(champion, challenger)
        self.assertTrue(verdict.passed)
        self.assertTrue(all(g["passed"] for g in verdict.gates))

    def test_pretty_ev_cannot_hide_bad_drawdown(self):
        champion = ModelScorecard("champ", 500, 6.0, 2.0, -0.05, 0.03, 0.65)
        challenger = ModelScorecard("chall", 500, 12.0, 3.0, -0.20, 0.03, 0.80)
        verdict = evaluate_challenger(champion, challenger)
        self.assertFalse(verdict.passed)
        failed = {g["name"] for g in verdict.gates if not g["passed"]}
        self.assertIn("drawdown_not_materially_worse", failed)


if __name__ == "__main__":
    unittest.main()
