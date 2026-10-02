"""Champion/challenger governance for GOLD 22 / Victory Sprint 11."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ChallengerPolicy:
    min_closed_trades: int = 100
    min_ev_improvement_bps: float = 1.0
    max_drawdown_degradation: float = 0.01
    max_ece_degradation: float = 0.005
    require_positive_stress_ev: bool = True


@dataclass(frozen=True)
class ModelScorecard:
    artifact_id: str
    closed_trades: int
    realized_ev_bps: float
    stressed_ev_bps: float
    max_drawdown: float
    ece: float
    positive_window_rate: float

    def __post_init__(self) -> None:
        if self.closed_trades < 0:
            raise ValueError("closed_trades must be non-negative")
        if not 0 <= self.positive_window_rate <= 1:
            raise ValueError("positive_window_rate must be in [0,1]")


@dataclass(frozen=True)
class ChallengerVerdict:
    champion_id: str
    challenger_id: str
    passed: bool
    gates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_challenger(champion: ModelScorecard, challenger: ModelScorecard, policy: ChallengerPolicy | None = None) -> ChallengerVerdict:
    policy = policy or ChallengerPolicy()
    gates = [
        {
            "name": "minimum_closed_trades",
            "passed": challenger.closed_trades >= policy.min_closed_trades,
            "value": challenger.closed_trades,
            "threshold": policy.min_closed_trades,
        },
        {
            "name": "realized_ev_improvement",
            "passed": challenger.realized_ev_bps >= champion.realized_ev_bps + policy.min_ev_improvement_bps,
            "value": challenger.realized_ev_bps - champion.realized_ev_bps,
            "threshold": policy.min_ev_improvement_bps,
        },
        {
            "name": "drawdown_not_materially_worse",
            "passed": challenger.max_drawdown >= champion.max_drawdown - policy.max_drawdown_degradation,
            "value": challenger.max_drawdown - champion.max_drawdown,
            "threshold": -policy.max_drawdown_degradation,
        },
        {
            "name": "calibration_not_materially_worse",
            "passed": challenger.ece <= champion.ece + policy.max_ece_degradation,
            "value": challenger.ece - champion.ece,
            "threshold": policy.max_ece_degradation,
        },
        {
            "name": "positive_window_rate_not_worse",
            "passed": challenger.positive_window_rate >= champion.positive_window_rate,
            "value": challenger.positive_window_rate - champion.positive_window_rate,
            "threshold": 0.0,
        },
    ]
    if policy.require_positive_stress_ev:
        gates.append({
            "name": "positive_stressed_ev",
            "passed": challenger.stressed_ev_bps > 0,
            "value": challenger.stressed_ev_bps,
            "threshold": ">0",
        })
    return ChallengerVerdict(
        champion_id=champion.artifact_id,
        challenger_id=challenger.artifact_id,
        passed=all(g["passed"] for g in gates),
        gates=tuple(gates),
    )


def rank_challengers(champion: ModelScorecard, challengers: list[ModelScorecard], policy: ChallengerPolicy | None = None) -> list[dict[str, Any]]:
    rows = []
    for challenger in challengers:
        verdict = evaluate_challenger(champion, challenger, policy)
        rows.append({
            "challenger": asdict(challenger),
            "verdict": verdict.to_dict(),
            "shadow_only": True,
            "promotion_eligible": verdict.passed,
        })
    return sorted(rows, key=lambda row: row["challenger"]["realized_ev_bps"], reverse=True)
