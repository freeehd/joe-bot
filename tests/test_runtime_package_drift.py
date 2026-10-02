import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from monitoring.drift import build_drift_baseline, score_feature_drift
from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.promotion import promote_runtime_package
from research.runtime_package import RuntimeComponent, RuntimePackageBuilder


class RuntimePackageDriftTests(unittest.TestCase):
    def test_drift_baseline_flags_shift(self):
        baseline_frame=pd.DataFrame({"a":list(range(100)),"b":[1.0]*100})
        baseline=build_drift_baseline(baseline_frame,["a","b"])
        same=score_feature_drift(baseline_frame,baseline)
        shifted=score_feature_drift(pd.DataFrame({"a":[1000+i for i in range(100)],"b":[1.0]*100}),baseline)
        self.assertEqual(same["status"],"OK")
        self.assertEqual(shifted["status"],"FAIL")

    def test_runtime_package_is_immutable_and_shadow_promotion_keeps_live_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); registry=ArtifactRegistry(root/"registry")
            parent_file=root/"validated.pkl"; parent_file.write_bytes(b"alpha")
            parent=registry.register(artifact_id="validated",artifact_type="validated-alpha",files=[parent_file],gates=[PromotionGate("core_edge",True)],repo_root=root)
            model=root/"model.pkl"; model.write_bytes(b"model")
            drift=root/"drift.json"; drift.write_text(json.dumps({"features":{}}))
            builder=RuntimePackageBuilder(registry_root=root/"registry",package_root=root/"packages")
            built=builder.build(package_id="runtime-1",parent_artifact_id="validated",components=[RuntimeComponent("model.pkl",model)],runtime_config={"paper_only":True},drift_baseline_path=drift,repo_root=root)
            self.assertFalse(built["manifest"]["live_capital_authorized"])
            with self.assertRaises(FileExistsError):
                builder.build(package_id="runtime-1",parent_artifact_id="validated",components=[RuntimeComponent("model.pkl",model)],runtime_config={},drift_baseline_path=drift,repo_root=root)
            path=promote_runtime_package(registry,"runtime-1",channel="shadow",repo_root=root)
            self.assertFalse(json.loads(path.read_text())["live_capital_authorized"])
            with self.assertRaises(ValueError):
                promote_runtime_package(registry,"runtime-1",channel="tiny-live",repo_root=root)


if __name__ == "__main__":
    unittest.main()
