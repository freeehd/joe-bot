"""Victory Sprint 6 governance: position telemetry, alpha decay, adaptive exits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.exit_intelligence import build_exit_intelligence_gate


def _eligible_parent(parent: dict) -> bool:
    if parent.get("artifact_type") == "safety-intelligence-challenger":
        gate = parent.get("metrics", {}).get("sprint5", {})
        return bool(gate.get("passed") and gate.get("verdict") == "PASS")
    if parent.get("artifact_type") == "intelligence-challenger":
        gate = parent.get("metrics", {}).get("intelligence_expansion", {})
        return bool(gate.get("passed") and gate.get("verdict") == "PASS")
    if parent.get("artifact_type") == "validated-alpha":
        gate = parent.get("metrics", {}).get("core_edge", {})
        return bool(gate.get("passed") and gate.get("verdict") == "PASS")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Victory Sprint 6 exit-intelligence evidence")
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--adaptive-metrics", required=True)
    parser.add_argument("--classifier-metrics", required=True)
    parser.add_argument("--alpha-decay", required=True)
    parser.add_argument("--excursion-report", required=True)
    parser.add_argument("--registry-root", default="data/registry")
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    parent = registry.read(args.parent_artifact)
    if not registry.verify(args.parent_artifact, repo_root=".")["valid"]:
        raise RuntimeError("parent artifact failed integrity verification")
    if not _eligible_parent(parent):
        raise RuntimeError("Sprint 6 requires an empirically passed upstream artifact")

    paths = [Path(args.baseline_metrics), Path(args.adaptive_metrics), Path(args.classifier_metrics), Path(args.alpha_decay), Path(args.excursion_report)]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    baseline, adaptive, classifier, alpha_decay, excursion = payloads
    gate = build_exit_intelligence_gate(
        baseline_metrics=baseline,
        adaptive_metrics=adaptive,
        classifier_metrics=classifier,
    )

    artifact_id = f"{args.parent_artifact}-sprint6-exit"
    artifact = registry.register(
        artifact_id=artifact_id,
        artifact_type="exit-intelligence-challenger" if gate["passed"] else "exit-intelligence-failure",
        files=paths,
        source_datasets=parent.get("source_datasets", []),
        parents=[args.parent_artifact],
        config={
            "deterministic_stops_remain_hard": True,
            "adaptive_authority": "early_exit_or_hold_only",
            "can_widen_stop": False,
            "can_increase_position": False,
        },
        metrics={"exit_intelligence": gate, "alpha_decay": alpha_decay, "excursion": excursion},
        gates=[PromotionGate(g["name"], g["passed"], g["value"], g["threshold"]) for g in gate["gates"]],
        status="candidate" if gate["passed"] else "rejected",
        repo_root=".",
        notes="Adaptive exit logic cannot widen hard stops, increase size, or bypass session flattening.",
    )
    print(json.dumps({
        "artifact_id": artifact_id,
        "manifest_sha256": artifact.digest,
        "verdict": gate["verdict"],
        "failed_gates": [g["name"] for g in gate["gates"] if not g["passed"]],
        "next_action": gate["promotion_instruction"],
    }, indent=2))


if __name__ == "__main__":
    main()
