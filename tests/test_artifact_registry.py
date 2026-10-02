import json
import tempfile
import unittest
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate


class ArtifactRegistryTests(unittest.TestCase):
    def test_register_verify_and_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_file = root / "model.bin"
            artifact_file.write_bytes(b"joe-bot-model")
            registry = ArtifactRegistry(root / "registry")
            manifest = registry.register(
                artifact_id="alpha-001",
                artifact_type="alpha-model",
                files=[artifact_file],
                gates=[PromotionGate("held_out_ev_positive", True, 6.2, ">0")],
            )
            stored = registry.read("alpha-001")
            self.assertEqual(stored["artifact_id"], "alpha-001")
            self.assertEqual(stored["manifest_sha256"], manifest.digest)
            self.assertTrue(registry.verify("alpha-001")["valid"])
            promoted = registry.promote("alpha-001", channel="shadow")
            self.assertEqual(json.loads(promoted.read_text())["artifact_id"], "alpha-001")

    def test_registry_is_immutable_and_failed_gate_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_file = root / "model.bin"
            artifact_file.write_bytes(b"v1")
            registry = ArtifactRegistry(root / "registry")
            registry.register(
                artifact_id="bad-001",
                artifact_type="alpha-model",
                files=[artifact_file],
                gates=[PromotionGate("held_out_ev_positive", False, -1.0, ">0")],
            )
            with self.assertRaises(FileExistsError):
                registry.register(artifact_id="bad-001", artifact_type="alpha-model", files=[artifact_file])
            with self.assertRaises(RuntimeError):
                registry.promote("bad-001")

    def test_verification_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_file = root / "model.bin"
            artifact_file.write_bytes(b"v1")
            registry = ArtifactRegistry(root / "registry")
            registry.register(artifact_id="alpha-001", artifact_type="alpha-model", files=[artifact_file])
            artifact_file.write_bytes(b"v2")
            self.assertFalse(registry.verify("alpha-001")["valid"])
            with self.assertRaises(RuntimeError):
                registry.promote("alpha-001")


if __name__ == "__main__":
    unittest.main()
