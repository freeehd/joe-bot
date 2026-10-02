import unittest
from research.victory_sprint6 import _eligible_parent


class VictorySprint6Tests(unittest.TestCase):
    def test_eligible_parent_requires_passed_upstream_evidence(self):
        self.assertTrue(_eligible_parent({"artifact_type":"validated-alpha","metrics":{"core_edge":{"passed":True,"verdict":"PASS"}}}))
        self.assertTrue(_eligible_parent({"artifact_type":"intelligence-challenger","metrics":{"intelligence_expansion":{"passed":True,"verdict":"PASS"}}}))
        self.assertTrue(_eligible_parent({"artifact_type":"safety-intelligence-challenger","metrics":{"sprint5":{"passed":True,"verdict":"PASS"}}}))
        self.assertFalse(_eligible_parent({"artifact_type":"validated-alpha","metrics":{"core_edge":{"passed":False,"verdict":"FAIL"}}}))


if __name__ == "__main__":
    unittest.main()
