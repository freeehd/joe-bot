"""Victory Sprint 4 orchestrator: regime + specialists + calibrated meta-model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.engine import ExecutionConfig
from models.train_v2 import load_dataset
from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.dataset_integrity import freeze_dataset
from research.intelligence_expansion import build_intelligence_expansion_gate, run_intelligence_expansion_walk_forward
from research.storage import ParquetDataLake
from research.walk_forward import WalkForwardConfig


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8")


def _core_edge_passed(artifact: dict) -> bool:
    if artifact.get("artifact_type") != "validated-alpha":
        return False
    metrics = artifact.get("metrics", {}).get("core_edge", {})
    return bool(metrics.get("passed") and metrics.get("verdict") == "PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Victory Sprint 4 intelligence expansion")
    parser.add_argument("--artifact-id", required=True, help="Sprint 3 validated-alpha artifact ID")
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.60)
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    parent = registry.read(args.artifact_id)
    if not registry.verify(args.artifact_id, repo_root=".")["valid"]:
        raise RuntimeError("parent validated-alpha artifact failed integrity verification")
    if not _core_edge_passed(parent):
        raise RuntimeError("Victory Sprint 4 requires a validated-alpha artifact with CORE EDGE PASS")
    if not parent.get("source_datasets"):
        raise RuntimeError("validated-alpha artifact has no source dataset lineage")

    source = parent["source_datasets"][0]
    dataset_version = source["dataset_version"]
    frozen = freeze_dataset(dataset_version, root=args.data_root)
    if frozen["dataset_fingerprint"] != source["dataset_fingerprint"]:
        raise RuntimeError("dataset fingerprint changed since Sprint 3")

    dataset, manifest, features = load_dataset(dataset_version, root=args.data_root)
    model_name = parent["config"]["model"]
    calibration = parent["config"]["calibration"]
    result = run_intelligence_expansion_walk_forward(
        dataset,
        manifest,
        features,
        lake=ParquetDataLake(args.data_root),
        model_name=model_name,
        calibration_method=calibration,
        walk_forward_config=WalkForwardConfig(confidence_threshold=args.confidence),
        execution_config=ExecutionConfig(
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            fee_bps=args.fee_bps,
            entry_delay_bars=args.entry_delay_bars,
        ),
    )
    gate = build_intelligence_expansion_gate(result)
    out = Path(args.data_root) / "experiments" / dataset_version / "sprint4" / args.artifact_id
    result_path = out / "intelligence_expansion_walk_forward.json"
    gate_path = out / "intelligence_expansion_gate.json"
    _write(result_path, result)
    _write(gate_path, gate)

    artifact_id = f"{args.artifact_id}-sprint4-intelligence"
    status = "candidate" if gate["passed"] else "rejected"
    artifact_type = "intelligence-challenger" if gate["passed"] else "intelligence-expansion-failure"
    artifact = registry.register(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        files=[result_path, gate_path],
        source_datasets=[source],
        parents=[args.artifact_id],
        config={
            "core_model": model_name,
            "core_calibration": calibration,
            "specialists": ["momentum", "breakout", "pullback", "mean_reversion"],
            "meta_model": "calibrated_logistic_regression",
            "selection_final_test_used": False,
            "execution": {
                "spread_bps": args.spread_bps,
                "slippage_bps": args.slippage_bps,
                "fee_bps": args.fee_bps,
                "entry_delay_bars": args.entry_delay_bars,
            },
        },
        metrics={"intelligence_expansion": gate},
        gates=[PromotionGate(g["name"], g["passed"], g["value"], g["threshold"]) for g in gate["gates"]],
        status=status,
        repo_root=".",
        notes="Sprint 4 pre-final-test intelligence expansion. PASS only permits one-time final-test audit; it does not authorize live capital.",
    )
    summary = {
        "parent_validated_alpha": args.artifact_id,
        "artifact_id": artifact_id,
        "artifact_manifest_sha256": artifact.digest,
        "verdict": gate["verdict"],
        "expectancy_improvement_bps": gate["expectancy_improvement_bps"],
        "failed_gates": [g["name"] for g in gate["gates"] if not g["passed"]],
        "next_action": gate["promotion_instruction"],
    }
    _write(out / "sprint4_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
