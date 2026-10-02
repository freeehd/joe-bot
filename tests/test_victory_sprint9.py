import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.victory_sprint9 import _promotion_artifact


class VictorySprint9Tests(unittest.TestCase):
    def test_promotion_lookup_reads_exact_promoted_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(Path(tmp) / "registry")
            registry.promotions_dir.mkdir(parents=True)
            (registry.promotions_dir / "shadow.json").write_text(json.dumps({"artifact_id": "runtime-1"}), encoding="utf-8")
            self.assertEqual(_promotion_artifact(registry, "shadow"), "runtime-1")
            self.assertIsNone(_promotion_artifact(registry, "paper"))

    def test_sprint9_proof_gate_does_not_authorize_live_capital(self):
        # The orchestrator records live-capital authorization as a deliberately failed gate.
        source = Path("research/victory_sprint9.py").read_text(encoding="utf-8")
        self.assertIn('PromotionGate("live_capital_authorized", False', source)
        self.assertIn('"live_capital_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
