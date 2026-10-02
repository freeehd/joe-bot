"""Victory Sprint 3 economic-proof diagnostics and explicit CORE EDGE gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoreEdgeThresholds:
    min_expectancy_bps: float = 0.0
    min_positive_window_rate: float = 0.60
    max_abs_drawdown: float = 0.15
    max_ece: float = 0.10
    min_stress_expectancy_bps: float = 0.0
    min_ev_spearman: float = 0.35
    min_ev_adjacent_monotonic_rate: float = 0.60
    max_single_symbol_trade_share: float = 0.35


def ev_monotonicity(
    trades: pd.DataFrame,
    *,
    ev_column: str = "net_ev_bps_at_entry",
    return_column: str = "net_return",
    buckets: int = 5,
) -> dict[str, Any]:
    required = {ev_column, return_column}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"trades missing EV diagnostic columns: {sorted(missing)}")
    frame = trades[[ev_column, return_column]].copy()
    frame[ev_column] = pd.to_numeric(frame[ev_column], errors="coerce")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < max(10, buckets * 2):
        return {"sufficient_data": False, "trades": int(len(frame)), "buckets": []}

    try:
        frame["ev_bucket"] = pd.qcut(frame[ev_column], q=buckets, duplicates="drop")
    except ValueError:
        return {"sufficient_data": False, "trades": int(len(frame)), "buckets": []}
    grouped = frame.groupby("ev_bucket", observed=True)
    rows = []
    for idx, (_, group) in enumerate(grouped, start=1):
        rows.append({
            "bucket": idx,
            "trades": int(len(group)),
            "predicted_ev_bps": float(group[ev_column].mean()),
            "realized_bps": float(group[return_column].mean() * 10_000.0),
        })
    predicted = pd.Series([row["predicted_ev_bps"] for row in rows], dtype=float)
    realized = pd.Series([row["realized_bps"] for row in rows], dtype=float)
    spearman = float(predicted.rank().corr(realized.rank())) if len(rows) >= 2 else float("nan")
    comparisons = [rows[i + 1]["realized_bps"] >= rows[i]["realized_bps"] for i in range(len(rows) - 1)]
    adjacent_rate = float(np.mean(comparisons)) if comparisons else 0.0
    return {
        "sufficient_data": True,
        "trades": int(len(frame)),
        "bucket_count": len(rows),
        "spearman": spearman,
        "adjacent_monotonic_rate": adjacent_rate,
        "buckets": rows,
    }


def symbol_concentration(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "symbol" not in trades.columns:
        return {"sufficient_data": False, "top_symbol_trade_share": 1.0, "by_symbol": []}
    counts = trades["symbol"].astype(str).value_counts()
    total = int(counts.sum())
    rows = [{"symbol": str(symbol), "trades": int(count), "trade_share": float(count / total)} for symbol, count in counts.items()]
    return {
        "sufficient_data": True,
        "symbols": int(len(counts)),
        "top_symbol_trade_share": float(counts.iloc[0] / total),
        "by_symbol": rows,
    }


def build_core_edge_report(
    *,
    classification_walk_forward: dict,
    v06_walk_forward: dict,
    v07_portfolio_walk_forward: dict,
    thresholds: CoreEdgeThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or CoreEdgeThresholds()
    portfolio_metrics = v07_portfolio_walk_forward["portfolio_trade_metrics"]
    records = pd.DataFrame(v07_portfolio_walk_forward.get("portfolio_trade_records", []))
    ev_diag = ev_monotonicity(records) if not records.empty else {"sufficient_data": False, "trades": 0, "buckets": []}
    concentration = symbol_concentration(records)

    stress = v06_walk_forward.get("execution_stress", {})
    stress_expectancies = {
        name: float(metrics.get("expectancy_bps", 0.0))
        for name, metrics in stress.items()
        if name != "baseline"
    }
    worst_stress = min(stress_expectancies.values()) if stress_expectancies else float("-inf")
    ece = float(classification_walk_forward["average_metrics"]["ece_10"])
    expectancy = float(portfolio_metrics.get("expectancy_bps", 0.0))
    positive_window_rate = float(v07_portfolio_walk_forward.get("portfolio_positive_return_window_rate", 0.0))
    worst_drawdown = float(v07_portfolio_walk_forward.get("portfolio_worst_window_drawdown", 0.0))

    def gate(name: str, passed: bool, value: Any, threshold: Any) -> dict[str, Any]:
        return {"name": name, "passed": bool(passed), "value": value, "threshold": threshold}

    gates = [
        gate("positive_net_oos_expectancy", expectancy > thresholds.min_expectancy_bps, expectancy, f">{thresholds.min_expectancy_bps}"),
        gate("walk_forward_stability", positive_window_rate >= thresholds.min_positive_window_rate, positive_window_rate, f">={thresholds.min_positive_window_rate}"),
        gate("useful_probability_calibration", ece <= thresholds.max_ece, ece, f"<={thresholds.max_ece}"),
        gate("acceptable_drawdown", worst_drawdown >= -thresholds.max_abs_drawdown, worst_drawdown, f">=-{thresholds.max_abs_drawdown}"),
        gate("positive_under_all_configured_cost_stress", bool(stress_expectancies) and worst_stress > thresholds.min_stress_expectancy_bps, worst_stress, f">{thresholds.min_stress_expectancy_bps}"),
        gate("ev_ranking_spearman", ev_diag.get("sufficient_data", False) and float(ev_diag.get("spearman", -1.0)) >= thresholds.min_ev_spearman, ev_diag.get("spearman"), f">={thresholds.min_ev_spearman}"),
        gate("ev_bucket_monotonicity", ev_diag.get("sufficient_data", False) and float(ev_diag.get("adjacent_monotonic_rate", 0.0)) >= thresholds.min_ev_adjacent_monotonic_rate, ev_diag.get("adjacent_monotonic_rate"), f">={thresholds.min_ev_adjacent_monotonic_rate}"),
        gate("no_single_symbol_dependency", concentration.get("sufficient_data", False) and float(concentration["top_symbol_trade_share"]) <= thresholds.max_single_symbol_trade_share, concentration.get("top_symbol_trade_share"), f"<={thresholds.max_single_symbol_trade_share}"),
    ]
    passed = all(item["passed"] for item in gates)
    return {
        "gate": "CORE EDGE",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": asdict(thresholds),
        "gates": gates,
        "classification": classification_walk_forward["average_metrics"],
        "v06": {
            "combined_trade_metrics": v06_walk_forward.get("combined_trade_metrics", {}),
            "positive_expectancy_window_rate": v06_walk_forward.get("positive_expectancy_window_rate"),
            "execution_stress": stress,
        },
        "v07": {
            "portfolio_trade_metrics": portfolio_metrics,
            "baseline_trade_metrics": v07_portfolio_walk_forward.get("baseline_trade_metrics", {}),
            "expectancy_improvement_bps": v07_portfolio_walk_forward.get("expectancy_improvement_bps"),
            "positive_return_window_rate": positive_window_rate,
            "worst_window_drawdown": worst_drawdown,
        },
        "ev_monotonicity": ev_diag,
        "symbol_concentration": concentration,
        "promotion_instruction": (
            "Eligible for the next research stage; still not authorization for live capital."
            if passed else
            "STOP promotion. Return to labels/features/models and investigate failed gates."
        ),
    }
