"""Victory Sprint 9 orchestrator: shadow + paper campaign proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.campaign_metrics import CampaignThresholds, evaluate_campaign_gate, summarize_paper_campaign, summarize_shadow_campaign
from research.fault_injection import run_fault_injection


def _promotion_artifact(registry: ArtifactRegistry, channel: str) -> str | None:
    path = registry.promotions_dir / f"{channel}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("artifact_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Victory Sprint 9 shadow/paper campaign")
    parser.add_argument("--runtime-package", required=True)
    parser.add_argument("--shadow-audit", action="append", required=True)
    parser.add_argument("--paper-audit", action="append", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--output", default="data/experiments/victory_sprint9.json")
    parser.add_argument("--min-shadow-sessions", type=int, default=5)
    parser.add_argument("--min-shadow-trades", type=int, default=100)
    parser.add_argument("--max-shadow-ev-drift-bps", type=float, default=5.0)
    parser.add_argument("--min-paper-sessions", type=int, default=10)
    parser.add_argument("--min-paper-trades", type=int, default=300)
    args = parser.parse_args()

    registry = ArtifactRegistry(args.registry_root)
    runtime = registry.read(args.runtime_package)
    if runtime.get("artifact_type") != "runtime-package":
        raise ValueError("Sprint 9 requires a Sprint 8 runtime-package")
    if not registry.verify(args.runtime_package, repo_root=".")["valid"]:
        raise RuntimeError("runtime package failed integrity verification")
    if _promotion_artifact(registry, "shadow") != args.runtime_package:
        raise RuntimeError("runtime package has not been explicitly promoted to shadow")
    if _promotion_artifact(registry, "paper") != args.runtime_package:
        raise RuntimeError("runtime package has not been explicitly promoted to paper")

    shadow = summarize_shadow_campaign(args.shadow_audit)
    paper = summarize_paper_campaign(args.paper_audit)
    fault = run_fault_injection()
    thresholds = CampaignThresholds(
        min_shadow_sessions=args.min_shadow_sessions,
        min_shadow_closed_trades=args.min_shadow_trades,
        max_abs_shadow_ev_drift_bps=args.max_shadow_ev_drift_bps,
        min_paper_sessions=args.min_paper_sessions,
        min_paper_closed_trades=args.min_paper_trades,
    )
    report = evaluate_campaign_gate(shadow, paper, fault, thresholds=thresholds)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    artifact_type = "shadow-paper-proof" if report["passed"] else "campaign-failure"
    artifact = registry.register(
        artifact_id=args.artifact_id,
        artifact_type=artifact_type,
        files=[path],
        source_datasets=runtime.get("source_datasets", []),
        parents=[args.runtime_package],
        config={"thresholds": report["thresholds"], "live_capital_authorized": False},
        metrics={
            "shadow_sessions": shadow["sessions"],
            "shadow_closed_trades": shadow["closed_trades"],
            "paper_sessions": paper["sessions"],
            "paper_closed_trades": paper["closed_trades"],
            "fault_injection_pass_rate": fault["pass_rate"],
        },
        gates=[
            PromotionGate("shadow_paper_campaign", report["passed"], report["verdict"], "PASS"),
            PromotionGate("live_capital_authorized", False, False, "remains false after Sprint 9"),
        ],
        status="candidate" if report["passed"] else "rejected",
        repo_root=".",
        notes="Sprint 9 operational proof. Passing this stage still does not authorize live capital.",
    )
    print(json.dumps({"report": report, "artifact_id": artifact.artifact_id, "artifact_type": artifact_type}, indent=2))


if __name__ == "__main__":
    main()
