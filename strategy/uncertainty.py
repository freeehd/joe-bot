"""Model-disagreement and uncertainty controls for Victory Sprint 5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class UncertaintyState:
    predictive_entropy: float
    directional_disagreement: float
    max_probability_gap: float
    risk_multiplier: float
    force_wait: bool


def normalized_entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("probabilities must be a 1D vector")
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("invalid probability vector")
    p = p / p.sum()
    return float(-np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))) / np.log(len(p)))


def disagreement_state(
    probability_vectors: Sequence[Sequence[float]],
    *,
    force_wait_threshold: float = 0.72,
    min_risk_multiplier: float = 0.15,
) -> UncertaintyState:
    matrix = np.asarray(probability_vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != 3:
        raise ValueError("expected N x 3 WAIT/LONG/SHORT probabilities")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("invalid probability matrix")
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    mean = matrix.mean(axis=0)
    entropy = normalized_entropy(mean)
    directional = matrix[:, 1] - matrix[:, 2]
    directional_disagreement = float(np.std(directional))
    winners = np.sort(mean)[::-1]
    gap = float(winners[0] - winners[1])
    composite = float(np.clip(0.65 * entropy + 0.35 * min(1.0, directional_disagreement * 2.5), 0.0, 1.0))
    force_wait = composite >= force_wait_threshold
    risk_multiplier = float(np.clip(1.0 - composite, min_risk_multiplier, 1.0))
    if force_wait:
        risk_multiplier = 0.0
    return UncertaintyState(entropy, directional_disagreement, gap, risk_multiplier, force_wait)
