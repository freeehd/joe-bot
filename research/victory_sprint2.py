"""Victory Sprint 2 orchestrator: ablate, benchmark, calibrate, walk-forward, freeze challengers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.alpha_tournament import (
    freeze_challenger,
    run_classification_walk_forward,
    run_feature_ablation,
    run_model_tournament,
)
from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.dataset_integrity import freeze_dataset
from models.train_v2 import load_dataset


def _write(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Victory Sprint 2 Core Alpha Tournament")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ablation-model", default="xgboost")
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--registry-root", default="data/registry")
    args = parser.parse_args()
    if args.finalists < 1:
        raise ValueError("--finalists must be >= 1")

    frozen = freeze_dataset(args.dataset_version, root=args.data_root)
    if not frozen["all_mandatory_gates_pass"]:
        failed = [g["name"] for g in frozen["gates"] if g["mandatory"] and not g["passed"]]
        raise RuntimeError(f"dataset failed mandatory gates: {failed}")

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    out = Path(args.data_root) / "experiments" / args.dataset_version / "sprint2"
    out.mkdir(parents=True, exist_ok=True)

    ablations = run_feature_ablation(dataset, manifest, features, model_name=args.ablation_model)
    _write(out / "feature_ablation.json", ablations)

    # Sprint 2 does not automatically drop feature groups merely because one
    # ablation wins one selection holdout.  Full V2 remains the tournament input;
    # ablations are evidence to revisit after Sprint 3 economics.
    tournament = run_model_tournament(dataset, manifest, features)
    _write(out / "model_tournament.json", tournament)

    finalists = tournament["results"][: args.finalists]
    registry = ArtifactRegistry(args.registry_root)
    frozen_rows = []
    for rank, finalist in enumerate(finalists, start=1):
        wf = run_classification_walk_forward(
            dataset,
            manifest,
            features,
            model_name=finalist["model"],
            calibration_method=finalist["calibration"],
        )
        wf_path = out / f"walk_forward_{finalist['candidate_id']}.json"
        _write(wf_path, wf)

        model_path = Path(args.data_root) / "models" / "victory" / args.dataset_version / f"{finalist['candidate_id']}.pkl"
        frozen_model = freeze_challenger(
            dataset,
            manifest,
            features,
            model_name=finalist["model"],
            calibration_method=finalist["calibration"],
            selection_rank=rank,
            selection_metrics=finalist["metrics"],
            output_path=model_path,
        )
        artifact_id = f"{args.dataset_version}-alpha-{rank:02d}-{finalist['candidate_id']}"
        manifest_record = registry.register(
            artifact_id=artifact_id,
            artifact_type="alpha-challenger",
            files=[model_path, frozen_model["metadata"], wf_path],
            source_datasets=[{
                "dataset_version": args.dataset_version,
                "dataset_fingerprint": frozen["dataset_fingerprint"],
            }],
            config={
                "model": finalist["model"],
                "calibration": finalist["calibration"],
                "feature_count": len(features),
                "selection_rank": rank,
            },
            metrics={
                "selection": finalist["metrics"],
                "pretest_walk_forward": wf["average_metrics"],
                "final_test_audit_only": frozen_model["final_test_metrics"],
            },
            gates=[
                PromotionGate("selection_did_not_use_final_test", True),
                PromotionGate("pretest_walk_forward_completed", wf["windows_completed"] > 0, wf["windows_completed"], ">0"),
                PromotionGate("final_test_not_used_for_ranking", True),
                PromotionGate("sprint3_economic_proof", False, None, "required before shadow promotion"),
            ],
            repo_root=".",
            notes="Frozen Sprint 2 challenger. Deliberately blocked from promotion until Sprint 3 economic proof.",
        )
        frozen_rows.append({
            "rank": rank,
            "artifact_id": artifact_id,
            "artifact_manifest_sha256": manifest_record.digest,
            "candidate": finalist["candidate_id"],
            "selection_score": finalist["selection_score"],
            "walk_forward_windows": wf["windows_completed"],
            "bundle": str(model_path),
        })

    summary = {
        "dataset_version": args.dataset_version,
        "dataset_fingerprint": frozen["dataset_fingerprint"],
        "feature_count": len(features),
        "ablation_experiments": len(ablations),
        "tournament_candidates": len(tournament["results"]),
        "unavailable_optional_models": tournament["unavailable_models"],
        "final_test_used_for_selection": False,
        "frozen_challengers": frozen_rows,
        "next_gate": "Victory Sprint 3 economic proof",
    }
    _write(out / "sprint2_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
