import tempfile
import unittest
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.go_live_review import review_go_live


class GoLiveReviewTests(unittest.TestCase):
    def _artifact(self, registry, root, artifact_id, artifact_type, passed=True):
        file = root / f"{artifact_id}.json"
        file.write_text("{}")
        registry.register(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            files=[file],
            gates=[PromotionGate("proof", passed)],
        )

    def test_full_prerequisites_make_experiment_eligible_but_not_live_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ArtifactRegistry(root / "registry")
            self._artifact(registry, root, "alpha", "validated-alpha")
            self._artifact(registry, root, "runtime", "runtime-package")
            self._artifact(registry, root, "campaign", "shadow-paper-proof")
            review = review_go_live(
                registry, ["alpha", "runtime", "campaign"],
                secret_hygiene_passed=True,
                recovery_test_passed=True,
                supervisor_test_passed=True,
                rbac_enabled=True,
                backup_restore_test_passed=True,
            )
            self.assertTrue(review.eligible_for_tiny_live_experiment)
            self.assertFalse(review.live_capital_authorized)

    def test_missing_or_failed_evidence_blocks_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ArtifactRegistry(root / "registry")
            self._artifact(registry, root, "alpha", "validated-alpha", passed=False)
            review = review_go_live(
                registry, ["alpha"],
                secret_hygiene_passed=True,
                recovery_test_passed=True,
                supervisor_test_passed=True,
                rbac_enabled=True,
                backup_restore_test_passed=True,
            )
            self.assertFalse(review.eligible_for_tiny_live_experiment)
            self.assertFalse(review.live_capital_authorized)


if __name__ == "__main__":
    unittest.main()
