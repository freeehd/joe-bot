"""Execution-policy economics and Sprint 7 promotion gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from backtest.micro_execution import FillSimulation


def summarize_fills(fills: Iterable[FillSimulation]) -> dict:
    rows = list(fills)
    if not rows:
        return {"orders": 0, "fill_rate": 0.0, "avg_cost_bps": 0.0, "p95_cost_bps": 0.0, "partial_rate": 0.0, "miss_rate": 1.0}
    costs = [row.cost_bps for row in rows if row.cost_bps is not None]
    return {
        "orders": len(rows),
        "fill_rate": float(np.mean([row.fill_rate for row in rows])),
        "avg_cost_bps": float(np.mean(costs)) if costs else 0.0,
        "p95_cost_bps": float(np.quantile(costs, .95)) if costs else 0.0,
        "partial_rate": float(np.mean([row.partial for row in rows])),
        "miss_rate": float(np.mean([row.missed for row in rows])),
    }


@dataclass(frozen=True)
class ExecutionPolicyThresholds:
    min_cost_improvement_bps: float = 0.25
    min_fill_rate: float = 0.90
    max_p95_cost_degradation_bps: float = 0.50


def build_execution_policy_gate(*, market: dict, challenger: dict, thresholds: ExecutionPolicyThresholds | None = None) -> dict:
    thresholds = thresholds or ExecutionPolicyThresholds()
    cost_improvement = float(market.get("avg_cost_bps", 0.0) - challenger.get("avg_cost_bps", 0.0))
    p95_degradation = float(challenger.get("p95_cost_bps", 0.0) - market.get("p95_cost_bps", 0.0))
    fill_rate = float(challenger.get("fill_rate", 0.0))
    gates = [
        {"name": "average_execution_cost_improves", "passed": cost_improvement >= thresholds.min_cost_improvement_bps, "value": cost_improvement, "threshold": f">={thresholds.min_cost_improvement_bps}"},
        {"name": "fill_rate_remains_high", "passed": fill_rate >= thresholds.min_fill_rate, "value": fill_rate, "threshold": f">={thresholds.min_fill_rate}"},
        {"name": "tail_cost_not_materially_worse", "passed": p95_degradation <= thresholds.max_p95_cost_degradation_bps, "value": p95_degradation, "threshold": f"<={thresholds.max_p95_cost_degradation_bps}"},
    ]
    passed = all(g["passed"] for g in gates)
    return {
        "gate": "EXECUTION POLICY",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": asdict(thresholds),
        "gates": gates,
        "cost_improvement_bps": cost_improvement,
        "promotion_instruction": (
            "Execution challenger may proceed to shadow A/B measurement."
            if passed else
            "Keep the existing execution policy; cheaper quotes are not useful if fill quality collapses."
        ),
    }
