"""Victory Sprint 8 orchestrator: build governed runtime package + drift baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.train_v2 import load_dataset
from monitoring.drift import build_drift_baseline
from research.artifact_registry import ArtifactRegistry
from research.runtime_package import RuntimeComponent, RuntimePackageBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Victory Sprint 8 governed runtime package")
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--package-root", default="data/runtime_packages")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--ev-state", required=True)
    parser.add_argument("--correlations", required=True)
    parser.add_argument("--risk-config", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--laya-calibration")
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    parent = registry.read(args.parent_artifact)
    if not registry.verify(args.parent_artifact, repo_root=".")["valid"]:
        raise RuntimeError("parent artifact failed integrity verification")
    if not parent.get("source_datasets"):
        raise RuntimeError("parent artifact has no source dataset lineage")
    source = parent["source_datasets"][0]
    dataset, _, features = load_dataset(source["dataset_version"], root=args.data_root)
    baseline = build_drift_baseline(dataset, features)
    baseline_path = Path(args.package_root) / f".{args.package_id}.drift-baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    components = [
        RuntimeComponent("alpha_model.pkl", args.model_bundle),
        RuntimeComponent("ev_state.json", args.ev_state),
        RuntimeComponent("correlations.csv", args.correlations),
        RuntimeComponent("risk_config.json", args.risk_config),
        RuntimeComponent("execution_config.json", args.execution_config),
    ]
    if args.laya_calibration:
        components.append(RuntimeComponent("laya_calibration.json", args.laya_calibration, required=False))
    runtime_config = {
        "dataset_version": source["dataset_version"],
        "dataset_fingerprint": source["dataset_fingerprint"],
        "feature_count": len(features),
        "paper_only": True,
        "live_capital_authorized": False,
    }
    builder = RuntimePackageBuilder(registry_root=args.registry_root, package_root=args.package_root)
    result = builder.build(
        package_id=args.package_id,
        parent_artifact_id=args.parent_artifact,
        components=components,
        runtime_config=runtime_config,
        drift_baseline_path=baseline_path,
        repo_root=".",
    )
    baseline_path.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
