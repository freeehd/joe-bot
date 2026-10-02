"""Sprint 5 diagnostics for model disagreement and OOD behavior."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from strategy.ood import RobustOODDetector
from strategy.uncertainty import disagreement_state


def annotate_ood(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    quantile: float = 0.995,
) -> tuple[pd.DataFrame, RobustOODDetector]:
    detector = RobustOODDetector(quantile=quantile).fit(train, calibration, feature_columns)
    result = test.copy()
    scores = detector.score_frame(test)
    result["ood_score"] = scores
    result["ood_threshold"] = detector.threshold
    result["is_ood"] = scores > detector.threshold
    result["ood_risk_multiplier"] = [detector.evaluate(row).risk_multiplier for _, row in test.iterrows()]
    return result, detector


def disagreement_table(probability_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    if not probability_sets:
        raise ValueError("probability_sets must not be empty")
    names = list(probability_sets)
    n = len(probability_sets[names[0]])
    if any(len(values) != n for values in probability_sets.values()):
        raise ValueError("all model probability arrays must have the same number of rows")
    rows = []
    for i in range(n):
        state = disagreement_state([probability_sets[name][i] for name in names])
        rows.append({
            "predictive_entropy": state.predictive_entropy,
            "directional_disagreement": state.directional_disagreement,
            "max_probability_gap": state.max_probability_gap,
            "uncertainty_risk_multiplier": state.risk_multiplier,
            "uncertainty_force_wait": state.force_wait,
        })
    return pd.DataFrame(rows)


def uncertainty_value_report(
    y_true: Sequence[int],
    probability_sets: dict[str, np.ndarray],
) -> dict:
    """Check whether disagreement/entropy actually identifies harder examples."""
    table = disagreement_table(probability_sets)
    arrays = [np.asarray(values, dtype=float) for values in probability_sets.values()]
    mean_probs = np.mean(np.stack(arrays, axis=0), axis=0)
    y = np.asarray(y_true, dtype=int)
    predicted = mean_probs.argmax(axis=1)
    table["correct"] = predicted == y
    table["uncertainty_score"] = 0.65 * table["predictive_entropy"] + 0.35 * np.minimum(1.0, table["directional_disagreement"] * 2.5)
    try:
        table["bucket"] = pd.qcut(table["uncertainty_score"], q=4, labels=False, duplicates="drop")
    except ValueError:
        return {"sufficient_data": False, "rows": int(len(table))}
    rows = []
    for bucket, group in table.groupby("bucket"):
        rows.append({
            "bucket": int(bucket),
            "rows": int(len(group)),
            "mean_uncertainty": float(group["uncertainty_score"].mean()),
            "accuracy": float(group["correct"].mean()),
        })
    rows.sort(key=lambda row: row["mean_uncertainty"])
    useful = bool(len(rows) >= 2 and rows[-1]["accuracy"] < rows[0]["accuracy"])
    return {
        "sufficient_data": len(rows) >= 2,
        "rows": int(len(table)),
        "high_uncertainty_accuracy_lower": useful,
        "accuracy_delta_high_minus_low": float(rows[-1]["accuracy"] - rows[0]["accuracy"]) if len(rows) >= 2 else 0.0,
        "buckets": rows,
    }


def ood_value_report(y_true: Sequence[int], probabilities: np.ndarray, ood_flags: Sequence[bool]) -> dict:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    flags = np.asarray(ood_flags, dtype=bool)
    if len(y) != len(probs) or len(y) != len(flags):
        raise ValueError("y/probabilities/ood_flags length mismatch")
    pred = probs.argmax(axis=1)
    correct = pred == y
    id_mask = ~flags
    if not id_mask.any() or not flags.any():
        return {"sufficient_data": False, "rows": int(len(y)), "ood_rate": float(flags.mean())}
    id_accuracy = float(correct[id_mask].mean())
    ood_accuracy = float(correct[flags].mean())
    return {
        "sufficient_data": True,
        "rows": int(len(y)),
        "ood_rate": float(flags.mean()),
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "ood_accuracy_delta": ood_accuracy - id_accuracy,
        "ood_is_harder": ood_accuracy < id_accuracy,
    }
