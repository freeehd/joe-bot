"""V0.8 Laya veto gate.

The gate is intentionally asymmetric: it may remove a quant-approved candidate,
but it cannot introduce a symbol, flip direction, increase size, or bypass the
V0.7 portfolio allocator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from models.laya_calibration import LayaCalibration
from models.laya_engine import LayaDecision, LayaEngine


STATE_FEATURES = (
    "return_1m", "return_3m", "return_5m", "return_10m", "return_15m",
    "return_acceleration_1m", "momentum_vs_mean_5", "momentum_vs_mean_15",
    "ema_5_20_separation", "ema_10_50_separation", "trend_persistence_10",
    "vwap_distance", "vwap_slope_5", "atr_14_norm", "realized_vol_5m",
    "realized_vol_15m", "range_expansion_20", "tod_relative_volume",
    "volume_acceleration_5", "spy_return_1m", "spy_return_5m", "spy_return_15m",
    "qqq_return_1m", "qqq_return_5m", "qqq_return_15m", "relative_strength_spy_5m",
    "relative_strength_qqq_5m", "breadth_up_1m", "minutes_since_open_norm",
    "minutes_until_close_norm",
)


@dataclass(frozen=True)
class LayaGateConfig:
    top_candidates: int = 5
    min_action_answer_confidence: float = 0.0
    risk_veto_probability: float = 0.70
    veto_poor_quality: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.top_candidates < 1:
            raise ValueError("top_candidates must be >= 1")
        for name in ("min_action_answer_confidence", "risk_veto_probability"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


def build_laya_state(
    candidate: dict,
    *,
    timestamp: pd.Timestamp | None = None,
    account_equity: float | None = None,
    current_positions: list[dict] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "symbol": str(candidate["symbol"]),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "proposed_direction": str(candidate["direction"]),
        "sector": str(candidate.get("sector", "UNKNOWN")),
        "p_wait": float(candidate.get("p_wait", 0.0)),
        "p_long": float(candidate.get("p_long", 0.0)),
        "p_short": float(candidate.get("p_short", 0.0)),
        "quant_confidence": float(candidate.get("confidence", 0.0)),
        "net_ev_bps": float(candidate.get("net_ev_bps", 0.0)),
        "structural_net_ev_bps": float(candidate.get("structural_net_ev", 0.0)) * 10_000.0,
        "empirical_samples": int(candidate.get("empirical_samples", 0)),
        "empirical_weight": float(candidate.get("empirical_weight", 0.0)),
        "max_directional_correlation": float(candidate.get("max_directional_correlation", 0.0)),
    }
    for feature in STATE_FEATURES:
        if feature in candidate and pd.notna(candidate[feature]):
            state[feature] = float(candidate[feature])

    positions = current_positions or []
    portfolio = {
        "open_positions": len(positions),
        "gross_exposure": float(sum(float(item.get("allocation", 0.0)) for item in positions)),
        "risk_dollars": float(sum(float(item.get("risk_dollars", 0.0)) for item in positions)),
        "long_exposure": float(sum(float(item.get("allocation", 0.0)) for item in positions if item.get("direction") == "LONG")),
        "short_exposure": float(sum(float(item.get("allocation", 0.0)) for item in positions if item.get("direction") == "SHORT")),
    }
    if account_equity is not None:
        portfolio["account_equity"] = float(account_equity)
    state["portfolio"] = portfolio
    return state


class LayaVetoGate:
    def __init__(
        self,
        engine: LayaEngine,
        *,
        config: LayaGateConfig | None = None,
        calibration: LayaCalibration | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or LayaGateConfig()
        self.calibration = calibration

    def _decision_reason(self, candidate: dict, decision: LayaDecision) -> str | None:
        proposed = str(candidate["direction"])
        if decision.action != proposed:
            return f"Laya action {decision.action} disagrees with quant {proposed}"
        if decision.action_answer_confidence < self.config.min_action_answer_confidence:
            return "Laya action answer confidence below calibrated threshold"
        if decision.risk_concern_probability >= self.config.risk_veto_probability:
            return "Laya risk concern above veto threshold"
        if self.config.veto_poor_quality and decision.quality == "POOR":
            return "Laya setup quality is POOR"
        return None

    def gate_candidates(
        self,
        ranked_candidates: list[dict],
        *,
        timestamp: pd.Timestamp | None = None,
        account_equity: float | None = None,
        current_positions: list[dict] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        # Pipeline rule: only the strongest N quant candidates reach Laya.
        candidates = [item.copy() for item in ranked_candidates[: self.config.top_candidates]]
        if not candidates:
            return {"approved": [], "vetoed": [], "decisions": [], "config": asdict(self.config)}

        states = [
            build_laya_state(
                candidate,
                timestamp=timestamp,
                account_equity=account_equity,
                current_positions=current_positions,
            )
            for candidate in candidates
        ]
        try:
            decisions = self.engine.evaluate_states(states)
        except Exception as exc:
            if not self.config.fail_closed:
                return {
                    "approved": candidates,
                    "vetoed": [],
                    "decisions": [],
                    "error": str(exc),
                    "config": asdict(self.config),
                }
            return {
                "approved": [],
                "vetoed": [
                    {"symbol": item["symbol"], "direction": item["direction"], "reason": f"Laya unavailable: {exc}"}
                    for item in candidates
                ],
                "decisions": [],
                "error": str(exc),
                "config": asdict(self.config),
            }

        approved: list[dict] = []
        vetoed: list[dict] = []
        decision_rows: list[dict] = []
        for candidate, raw_decision in zip(candidates, decisions):
            decision = self.calibration.calibrate(raw_decision) if self.calibration is not None else raw_decision
            reason = self._decision_reason(candidate, decision)
            annotated = candidate.copy()
            annotated.update(
                {
                    "laya_action": decision.action,
                    "laya_action_confidence": decision.action_answer_confidence,
                    "laya_quality": decision.quality,
                    "laya_quality_confidence": decision.quality_answer_confidence,
                    "laya_risk_concern_probability": decision.risk_concern_probability,
                    "laya_model": decision.model,
                    "laya_vetoed": reason is not None,
                }
            )
            decision_rows.append({"symbol": candidate["symbol"], **decision.to_dict(), "veto_reason": reason})
            if reason is None:
                approved.append(annotated)
            else:
                vetoed.append(
                    {
                        "symbol": candidate["symbol"],
                        "direction": candidate["direction"],
                        "reason": reason,
                        "decision": decision.to_dict(),
                    }
                )

        # Invariant: every output trade is a subset of the quant-approved input;
        # no candidate direction is modified by Laya.
        return {
            "approved": approved,
            "vetoed": vetoed,
            "decisions": decision_rows,
            "config": asdict(self.config),
        }
