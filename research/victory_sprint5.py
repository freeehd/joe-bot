"""Victory Sprint 5 governance: Laya + disagreement + OOD evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.sprint5_gate import build_sprint5_gate


def _parent_is_eligible(parent: dict) -> bool:
    if parent.get("artifact_type") == "validated-alpha":
        core = parent.get("metrics", {}).get("core_edge", {})
        return bool(core.get("passed") and core.get("verdict") == "PASS")
    if parent.get("artifact_type") == "intelligence-challenger":
        gate = parent.get("metrics", {}).get("intelligence_expansion", {})
        return bool(gate.get("passed") and gate.get("verdict") == "PASS")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Victory Sprint 5 additive intelligence/safety proof")
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--laya-result", required=True)
    parser.add_argument("--uncertainty-report", required=True)
    parser.add_argument("--ood-report", required=True)
    parser.add_argument("--registry-root", default="data/registry")
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    parent = registry.read(args.parent_artifact)
    if not registry.verify(args.parent_artifact, repo_root=".")["valid"]:
        raise RuntimeError("parent artifact failed integrity verification")
    if not _parent_is_eligible(parent):
        raise RuntimeError("Sprint 5 requires a CORE EDGE PASS or INTELLIGENCE EXPANSION PASS parent")

    laya_path = Path(args.laya_result)
    uncertainty_path = Path(args.uncertainty_report)
    ood_path = Path(args.ood_report)
    laya = json.loads(laya_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    ood = json.loads(ood_path.read_text(encoding="utf-8"))
    gate = build_sprint5_gate(laya_result=laya, uncertainty_report=uncertainty, ood_report=ood)

    artifact_id = f"{args.parent_artifact}-sprint5-safety"
    artifact = registry.register(
        artifact_id=artifact_id,
        artifact_type="safety-intelligence-challenger" if gate["passed"] else "sprint5-failure",
        files=[laya_path, uncertainty_path, ood_path],
        source_datasets=parent.get("source_datasets", []),
        parents=[args.parent_artifact],
        config={
            "laya_authority": "veto_only",
            "uncertainty_authority": "reduce_or_wait_only",
            "ood_authority": "reduce_or_veto_only",
        },
        metrics={"sprint5": gate},
        gates=[PromotionGate(g["name"], g["passed"], g.get("value"), g.get("threshold")) for g in gate["gates"]],
        status="candidate" if gate["passed"] else "rejected",
        repo_root=".",
        notes="Sprint 5 components remain asymmetric: they may reduce/veto authority, never create/flip/increase a trade.",
    )
    summary = {
        "artifact_id": artifact_id,
        "manifest_sha256": artifact.digest,
        "verdict": gate["verdict"],
        "failed_gates": [g["name"] for g in gate["gates"] if not g["passed"]],
        "next_action": gate["promotion_instruction"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
