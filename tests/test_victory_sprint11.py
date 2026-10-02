import tempfile
import unittest

from research.champion_challenger import ModelScorecard
from research.victory_sprint11 import run_sprint11


class VictorySprint11Tests(unittest.TestCase):
    def test_sprint11_never_authorizes_live_capital(self):
        with tempfile.TemporaryDirectory() as tmp:
            champion=ModelScorecard("champ",500,5,1,-.05,.03,.6)
            challenger=ModelScorecard("chall",500,8,2,-.05,.03,.7)
            result=run_sprint11(champion,challenger,registry_root=f"{tmp}/registry",output_dir=f"{tmp}/out")
            self.assertTrue(result["verdict"]["passed"])
            self.assertFalse(result["live_capital_authorized"])
            self.assertEqual(result["challenger_authority"],"shadow_only")


if __name__ == "__main__":
    unittest.main()
