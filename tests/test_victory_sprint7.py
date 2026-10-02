import unittest
from research.victory_sprint7 import _eligible_parent


class VictorySprint7Tests(unittest.TestCase):
    def test_parent_must_have_passed_empirical_gate(self):
        self.assertTrue(_eligible_parent({"artifact_type":"validated-alpha","metrics":{"core_edge":{"passed":True,"verdict":"PASS"}}}))
        self.assertTrue(_eligible_parent({"artifact_type":"exit-intelligence-challenger","metrics":{"exit_intelligence":{"passed":True,"verdict":"PASS"}}}))
        self.assertFalse(_eligible_parent({"artifact_type":"exit-intelligence-challenger","metrics":{"exit_intelligence":{"passed":False,"verdict":"FAIL"}}}))


if __name__ == "__main__":
    unittest.main()
