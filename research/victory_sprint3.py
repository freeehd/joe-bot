"""Victory Sprint 3 orchestrator: realistic economics, EV proof, CORE EDGE gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.engine import ExecutionConfig
from models.train_v2 import load_dataset
from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.core_edge import CoreEdgeThresholds, build_core_edge_report
from research.dataset_integrity import freeze_dataset
from research.portfolio_walk_forward import run_portfolio_walk_forward
from research.storage import ParquetDataLake
from research.walk_forward import WalkForwardConfig, run_walk_forward


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Victory Sprint 3 economic proof for one frozen challenger")
    parser.add_argument("--artifact-id", required=True, help="Sprint 2 alpha challenger artifact ID")
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    challenger = registry.read(args.artifact_id)
    verification = registry.verify(args.artifact_id, repo_root=".")
    if not verification["valid"]:
        raise RuntimeError("challenger artifact files failed integrity verification")
    if challenger.get("artifact_type") != "alpha-challenger":
        raise ValueError("--artifact-id must refer to a Sprint 2 alpha-challenger")
    if not challenger.get("source_datasets"):
        raise ValueError("challenger has no source dataset lineage")

    source = challenger["source_datasets"][0]
    dataset_version = source["dataset_version"]
    frozen = freeze_dataset(dataset_version, root=args.data_root)
    if frozen["dataset_fingerprint"] != source["dataset_fingerprint"]:
        raise RuntimeError("challenger dataset fingerprint no longer matches frozen dataset")

    dataset, manifest, features = load_dataset(dataset_version, root=args.data_root)
    config = challenger["config"]
    model_name = config["model"]
    calibration = config["calibration"]
    lake = ParquetDataLake(args.data_root)
    execution = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
        entry_delay_bars=args.entry_delay_bars,
    )
    wf_config = WalkForwardConfig(confidence_threshold=args.confidence)
    out = Path(args.data_root) / "experiments" / dataset_version / "sprint3" / args.artifact_id

    v06 = run_walk_forward(
        dataset,
        manifest,
        features,
        lake=lake,
        model_name=model_name,
        config=wf_config,
        execution_config=execution,
        calibration_method=calibration,
        stress_execution=True,
    )
    _write(out / "v06_walk_forward.json", v06)

    v07 = run_portfolio_walk_forward(
        dataset,
        manifest,
        features,
        lake=lake,
        model_name=model_name,
        walk_forward_config=wf_config,
        execution_config=execution,
        initial_equity=args.initial_equity,
        calibration_method=calibration,
    )
    _write(out / "v07_portfolio_walk_forward.json", v07)

    # The Sprint 2 artifact already contains pre-final-test classification WF
    # diagnostics. Reuse that immutable evidence instead of reselecting anything.
    pretest_wf = None
    for file_record in challenger["files"]:
        if "walk_forward_" in file_record["path"] and file_record["path"].endswith(".json"):
            pretest_wf = json.loads(Path(file_record["path"]).read_text(encoding="utf-8"))
            break
    if pretest_wf is None:
        raise RuntimeError("challenger artifact is missing pre-final-test classification walk-forward evidence")

    report = build_core_edge_report(
        classification_walk_forward=pretest_wf,
        v06_walk_forward=v06,
        v07_portfolio_walk_forward=v07,
        thresholds=CoreEdgeThresholds(),
    )
    report.update({
        "challenger_artifact_id": args.artifact_id,
        "dataset_version": dataset_version,
        "dataset_fingerprint": frozen["dataset_fingerprint"],
        "model": model_name,
        "calibration": calibration,
    })
    report_path = out / "core_edge_report.json"
    _write(report_path, report)

    validated_id = f"{args.artifact_id}-core-edge"
    model_files = [Path(item["path"]) for item in challenger["files"] if item["path"].endswith((".pkl", ".metadata.json"))]
    status = "candidate" if report["passed"] else "rejected"
    proof = registry.register(
        artifact_id=validated_id,
        artifact_type="validated-alpha" if report["passed"] else "core-edge-failure",
        files=[*model_files, out / "v06_walk_forward.json", out / "v07_portfolio_walk_forward.json", report_path],
        source_datasets=[source],
        parents=[args.artifact_id],
        config={
            "model": model_name,
            "calibration": calibration,
            "execution": {
                "spread_bps": args.spread_bps,
                "slippage_bps": args.slippage_bps,
                "fee_bps": args.fee_bps,
                "entry_delay_bars": args.entry_delay_bars,
            },
        },
        metrics={"core_edge": report},
        gates=[PromotionGate(item["name"], item["passed"], item["value"], item["threshold"]) for item in report["gates"]],
        status=status,
        repo_root=".",
        notes="Sprint 3 CORE EDGE proof. PASS permits later research stages, not live-capital deployment.",
    )

    summary = {
        "challenger": args.artifact_id,
        "proof_artifact": validated_id,
        "proof_manifest_sha256": proof.digest,
        "core_edge_verdict": report["verdict"],
        "failed_gates": [item["name"] for item in report["gates"] if not item["passed"]],
        "next_action": report["promotion_instruction"],
    }
    _write(out / "sprint3_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
