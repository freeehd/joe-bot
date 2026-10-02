"""Probabilistic market-regime inference from Feature Engine V2 context.

The regime engine is intentionally transparent and deterministic.  It produces a
probability distribution rather than a brittle hard label so downstream strategy
weights can respond smoothly to changing market conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

REGIMES = (
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "HIGH_VOL",
    "LOW_VOL",
    "SHOCK",
    "OPENING_VOLATILITY",
)


def _safe(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    values = np.array([float(scores[name]) for name in REGIMES], dtype=float) / temperature
    values -= values.max()
    exp = np.exp(values)
    probs = exp / exp.sum()
    return {name: float(prob) for name, prob in zip(REGIMES, probs)}


@dataclass(frozen=True)
class RegimeState:
    label: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    def to_dict(self) -> dict:
        return {
            "regime": self.label,
            "regime_confidence": self.confidence,
            "regime_uncertainty": self.uncertainty,
            "regime_probabilities": dict(self.probabilities),
        }


class RegimeEngine:
    def __init__(self, *, temperature: float = 0.85) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def score_row(self, row: Mapping[str, float]) -> dict[str, float]:
        spy5 = _safe(row, "spy_return_5m")
        spy15 = _safe(row, "spy_return_15m")
        qqq5 = _safe(row, "qqq_return_5m")
        qqq15 = _safe(row, "qqq_return_15m")
        breadth = np.clip(_safe(row, "breadth_up_1m", 0.5), 0.0, 1.0)
        vol5 = max(_safe(row, "realized_vol_5m"), 0.0)
        vol15 = max(_safe(row, "realized_vol_15m"), 0.0)
        range_exp = max(_safe(row, "range_expansion_20", 1.0), 0.0)
        rvol = max(_safe(row, "tod_relative_volume", 1.0), 0.0)
        opening = max(_safe(row, "opening_15m_flag"), _safe(row, "opening_hour_flag") * 0.5)

        market5 = 0.5 * (spy5 + qqq5)
        market15 = 0.5 * (spy15 + qqq15)
        trend = 55.0 * market5 + 28.0 * market15 + 1.2 * (breadth - 0.5)
        vol_ratio = vol5 / max(vol15, 1e-6)
        vol_score = 65.0 * vol5 + 0.65 * max(vol_ratio - 0.8, 0.0) + 0.35 * max(range_exp - 1.0, 0.0)
        shock = 95.0 * abs(market5) + 75.0 * vol5 + 0.55 * max(range_exp - 1.4, 0.0) + 0.20 * max(rvol - 1.7, 0.0)
        quiet = 1.0 - min(1.0, 80.0 * vol5 + 0.45 * max(range_exp - 0.8, 0.0))
        range_score = 1.1 - min(1.1, abs(trend)) + 0.30 * (1.0 - abs(breadth - 0.5) * 2.0)

        return {
            "TREND_UP": trend,
            "TREND_DOWN": -trend,
            "RANGE": range_score,
            "HIGH_VOL": vol_score,
            "LOW_VOL": quiet,
            "SHOCK": shock,
            "OPENING_VOLATILITY": 0.75 * opening + 0.35 * vol_score + 0.15 * max(rvol - 1.0, 0.0),
        }

    def infer_row(self, row: Mapping[str, float]) -> RegimeState:
        probabilities = _softmax(self.score_row(row), self.temperature)
        label = max(probabilities, key=probabilities.get)
        return RegimeState(label, probabilities[label], probabilities)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            state = self.infer_row(row)
            payload = state.to_dict()
            for regime, probability in state.probabilities.items():
                payload[f"regime_p_{regime.lower()}"] = probability
            rows.append(payload)
        return pd.DataFrame(rows, index=frame.index)
