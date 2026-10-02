import unittest

from research.victory_sprint5 import _parent_is_eligible


class VictorySprint5Tests(unittest.TestCase):
    def test_parent_can_be_validated_core_or_passed_intelligence(self):
        self.assertTrue(_parent_is_eligible({
            "artifact_type": "validated-alpha",
            "metrics": {"core_edge": {"passed": True, "verdict": "PASS"}},
        }))
        self.assertTrue(_parent_is_eligible({
            "artifact_type": "intelligence-challenger",
            "metrics": {"intelligence_expansion": {"passed": True, "verdict": "PASS"}},
        }))
        self.assertFalse(_parent_is_eligible({
            "artifact_type": "intelligence-challenger",
            "metrics": {"intelligence_expansion": {"passed": False, "verdict": "FAIL"}},
        }))


if __name__ == "__main__":
    unittest.main()
