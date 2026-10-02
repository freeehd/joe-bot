"""Victory Sprint 11 governance: champion/challenger + dynamic universe + events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry, PromotionGate
from research.champion_challenger import ChallengerPolicy, ModelScorecard, evaluate_challenger


def _scorecard(payload: dict) -> ModelScorecard:
    return ModelScorecard(**payload)


def run_sprint11(
    champion: ModelScorecard,
    challenger: ModelScorecard,
    *,
    registry_root: str = "data/registry",
    output_dir: str = "data/experiments/sprint11",
    policy: ChallengerPolicy | None = None,
) -> dict:
    verdict = evaluate_challenger(champion, challenger, policy)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{challenger.artifact_id}_vs_{champion.artifact_id}.json"
    result = {
        "victory_sprint": 11,
        "champion": champion.__dict__,
        "challenger": challenger.__dict__,
        "verdict": verdict.to_dict(),
        "challenger_authority": "shadow_only",
        "live_capital_authorized": False,
        "dynamic_universe_engineered": True,
        "structured_event_intelligence_engineered": True,
    }
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    registry = ArtifactRegistry(registry_root)
    artifact_id = f"sprint11-{challenger.artifact_id}-vs-{champion.artifact_id}"
    try:
        manifest = registry.register(
            artifact_id=artifact_id,
            artifact_type="champion-challenger-evaluation",
            files=[report_path],
            parents=[champion.artifact_id, challenger.artifact_id],
            gates=[
                PromotionGate("challenger_beats_champion", verdict.passed, details="all predefined challenger gates must pass"),
                PromotionGate("tiny_live_authorization", False, details="Sprint 11 never grants live-capital authority"),
            ],
            metrics={"challenger_realized_ev_bps": challenger.realized_ev_bps, "champion_realized_ev_bps": champion.realized_ev_bps},
        )
        result["registry_artifact_id"] = manifest.artifact_id
    except FileExistsError:
        result["registry_artifact_id"] = artifact_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GOLD 22 champion/challenger pair")
    parser.add_argument("--champion", required=True, help="Champion scorecard JSON")
    parser.add_argument("--challenger", required=True, help="Challenger scorecard JSON")
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--output-dir", default="data/experiments/sprint11")
    args = parser.parse_args()
    champion = _scorecard(json.loads(Path(args.champion).read_text(encoding="utf-8")))
    challenger = _scorecard(json.loads(Path(args.challenger).read_text(encoding="utf-8")))
    print(json.dumps(run_sprint11(champion, challenger, registry_root=args.registry_root, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
