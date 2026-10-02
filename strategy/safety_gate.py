"""Composable uncertainty/OOD safety authority used after alpha approval."""
from __future__ import annotations

from dataclasses import dataclass

from strategy.ood import OODResult
from strategy.uncertainty import UncertaintyState


@dataclass(frozen=True)
class SafetyDecision:
    approved: bool
    risk_multiplier: float
    reason: str


class MetaSafetyGate:
    """May reduce or veto an approved trade; never creates or reverses one."""

    def evaluate(self, uncertainty: UncertaintyState, ood: OODResult | None = None) -> SafetyDecision:
        if uncertainty.force_wait:
            return SafetyDecision(False, 0.0, "model disagreement/entropy forced WAIT")
        multiplier = uncertainty.risk_multiplier
        reasons = []
        if multiplier < 1.0:
            reasons.append("uncertainty reduced risk")
        if ood is not None:
            multiplier *= ood.risk_multiplier
            if ood.risk_multiplier == 0.0:
                return SafetyDecision(False, 0.0, "hard OOD veto")
            if ood.is_ood:
                reasons.append("OOD reduced risk")
        multiplier = max(0.0, min(1.0, multiplier))
        return SafetyDecision(multiplier > 0.0, multiplier, "; ".join(reasons) or "safety checks passed")
