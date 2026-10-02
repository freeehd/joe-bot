"""Victory Sprint 5 evidence gate for Laya, uncertainty, and OOD."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Sprint5Thresholds:
    min_laya_expectancy_delta_bps: float = 0.0
    max_laya_drawdown_degradation: float = 0.005
    require_uncertainty_harder_high_bucket: bool = True
    require_ood_harder: bool = True


def build_sprint5_gate(
    *,
    laya_result: dict,
    uncertainty_report: dict,
    ood_report: dict,
    thresholds: Sprint5Thresholds | None = None,
) -> dict:
    thresholds = thresholds or Sprint5Thresholds()
    gated = laya_result.get("gated_portfolio", {})
    laya_delta = float(gated.get("expectancy_delta_vs_ungated_bps", float("-inf")))
    dd_delta = float(gated.get("worst_drawdown_delta_vs_ungated", float("-inf")))
    uncertainty_useful = bool(uncertainty_report.get("sufficient_data") and uncertainty_report.get("high_uncertainty_accuracy_lower"))
    ood_useful = bool(ood_report.get("sufficient_data") and ood_report.get("ood_is_harder"))
    gates = [
        {"name": "laya_adds_expectancy", "passed": laya_delta > thresholds.min_laya_expectancy_delta_bps, "value": laya_delta, "threshold": f">{thresholds.min_laya_expectancy_delta_bps}"},
        {"name": "laya_drawdown_not_materially_worse", "passed": dd_delta >= -thresholds.max_laya_drawdown_degradation, "value": dd_delta, "threshold": f">=-{thresholds.max_laya_drawdown_degradation}"},
        {"name": "uncertainty_identifies_harder_states", "passed": uncertainty_useful if thresholds.require_uncertainty_harder_high_bucket else True, "value": uncertainty_report.get("accuracy_delta_high_minus_low"), "threshold": "high uncertainty accuracy < low uncertainty accuracy"},
        {"name": "ood_identifies_harder_states", "passed": ood_useful if thresholds.require_ood_harder else True, "value": ood_report.get("ood_accuracy_delta"), "threshold": "OOD accuracy < in-distribution accuracy"},
    ]
    passed = all(g["passed"] for g in gates)
    return {
        "gate": "LAYA + UNCERTAINTY/OOD",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": asdict(thresholds),
        "gates": gates,
        "promotion_instruction": (
            "Eligible to combine only the additive Sprint 5 safety/intelligence components."
            if passed else
            "Do not grant Sprint 5 components production authority; keep them diagnostic or reduce authority."
        ),
    }
