"""Victory Sprint 7 governance for microstructure and execution-policy challengers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.execution_policy_benchmark import build_execution_policy_gate


def _eligible_parent(parent: dict) -> bool:
    for artifact_type, metric_key in (
        ("exit-intelligence-challenger", "exit_intelligence"),
        ("safety-intelligence-challenger", "sprint5"),
        ("intelligence-challenger", "intelligence_expansion"),
        ("validated-alpha", "core_edge"),
    ):
        if parent.get("artifact_type") == artifact_type:
            metric = parent.get("metrics", {}).get(metric_key, {})
            return bool(metric.get("passed") and metric.get("verdict") == "PASS")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Victory Sprint 7 execution-policy evidence")
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--market-baseline", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--microstructure-manifest", required=True)
    parser.add_argument("--registry-root", default="data/registry")
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    parent = registry.read(args.parent_artifact)
    if not registry.verify(args.parent_artifact, repo_root=".")["valid"]:
        raise RuntimeError("parent artifact failed integrity verification")
    if not _eligible_parent(parent):
        raise RuntimeError("Sprint 7 requires an empirically passed upstream artifact")

    market_path = Path(args.market_baseline)
    challenger_path = Path(args.challenger)
    micro_path = Path(args.microstructure_manifest)
    market = json.loads(market_path.read_text(encoding="utf-8"))
    challenger = json.loads(challenger_path.read_text(encoding="utf-8"))
    gate = build_execution_policy_gate(market=market, challenger=challenger)

    artifact_id = f"{args.parent_artifact}-sprint7-execution"
    artifact = registry.register(
        artifact_id=artifact_id,
        artifact_type="execution-policy-challenger" if gate["passed"] else "execution-policy-failure",
        files=[market_path, challenger_path, micro_path],
        source_datasets=parent.get("source_datasets", []),
        parents=[args.parent_artifact],
        config={
            "microstructure_required": True,
            "partial_fills_modeled": True,
            "missed_fills_modeled": True,
            "shadow_ab_required_before_paper": True,
        },
        metrics={"execution_policy": gate},
        gates=[PromotionGate(g["name"], g["passed"], g["value"], g["threshold"]) for g in gate["gates"]],
        status="candidate" if gate["passed"] else "rejected",
        repo_root=".",
        notes="Execution challenger must still survive live shadow A/B before paper authority.",
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
