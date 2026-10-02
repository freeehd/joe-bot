import unittest

from research.victory_sprint4 import _core_edge_passed


class VictorySprint4Tests(unittest.TestCase):
    def test_requires_validated_alpha_with_core_edge_pass(self):
        self.assertTrue(_core_edge_passed({
            "artifact_type": "validated-alpha",
            "metrics": {"core_edge": {"passed": True, "verdict": "PASS"}},
        }))
        self.assertFalse(_core_edge_passed({
            "artifact_type": "alpha-challenger",
            "metrics": {"core_edge": {"passed": True, "verdict": "PASS"}},
        }))
        self.assertFalse(_core_edge_passed({
            "artifact_type": "validated-alpha",
            "metrics": {"core_edge": {"passed": False, "verdict": "FAIL"}},
        }))


if __name__ == "__main__":
    unittest.main()
