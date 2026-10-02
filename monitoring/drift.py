"""Feature/prediction drift baselines and PSI diagnostics for promoted artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftThresholds:
    warn_psi: float = 0.10
    fail_psi: float = 0.25


def _quantile_edges(series: pd.Series, bins: int) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return [-1.0, 1.0]
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(clean, quantiles)).astype(float)
    if len(edges) < 2:
        value = float(clean.iloc[0])
        edges = np.array([value - 1e-9, value + 1e-9])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(v) for v in edges]


def build_drift_baseline(frame: pd.DataFrame, feature_columns: list[str], *, bins: int = 10) -> dict[str, Any]:
    if bins < 3:
        raise ValueError("bins must be >= 3")
    features = {}
    for column in feature_columns:
        if column not in frame.columns:
            raise ValueError(f"missing drift feature {column!r}")
        clean = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        edges = _quantile_edges(clean, bins)
        counts, _ = np.histogram(clean.to_numpy(dtype=float), bins=np.asarray(edges, dtype=float))
        proportions = counts / max(counts.sum(), 1)
        features[column] = {
            "mean": float(clean.mean()) if len(clean) else 0.0,
            "std": float(clean.std(ddof=0)) if len(clean) else 0.0,
            "missing_rate": float(1.0 - len(clean) / max(len(frame), 1)),
            "bin_edges": edges,
            "bin_proportions": [float(v) for v in proportions],
        }
    return {"rows": int(len(frame)), "feature_count": len(feature_columns), "features": features}


def population_stability_index(expected: list[float], actual: list[float], *, epsilon: float = 1e-6) -> float:
    if len(expected) != len(actual):
        raise ValueError("expected/actual bins must match")
    e = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    a = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    e = e / e.sum(); a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def score_feature_drift(frame: pd.DataFrame, baseline: dict, thresholds: DriftThresholds | None = None) -> dict:
    thresholds = thresholds or DriftThresholds()
    rows = []
    for column, spec in baseline["features"].items():
        clean = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        edges = np.asarray(spec["bin_edges"], dtype=float)
        counts, _ = np.histogram(clean.to_numpy(dtype=float), bins=edges)
        actual = counts / max(counts.sum(), 1)
        psi = population_stability_index(spec["bin_proportions"], actual.tolist())
        status = "FAIL" if psi >= thresholds.fail_psi else "WARN" if psi >= thresholds.warn_psi else "OK"
        rows.append({"feature": column, "psi": psi, "status": status})
    worst = max((row["psi"] for row in rows), default=0.0)
    return {
        "features": rows,
        "worst_psi": worst,
        "failed_features": [r["feature"] for r in rows if r["status"] == "FAIL"],
        "warn_features": [r["feature"] for r in rows if r["status"] == "WARN"],
        "status": "FAIL" if any(r["status"] == "FAIL" for r in rows) else "WARN" if any(r["status"] == "WARN" for r in rows) else "OK",
    }
