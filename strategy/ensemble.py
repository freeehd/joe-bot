"""Regime-aware specialist ensemble/meta-policy for Sprint 4 research."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from strategy.regime import RegimeState
from strategy.specialists import SpecialistSignal

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "TREND_UP": {"momentum": 1.3, "breakout": 1.1, "pullback": 1.25, "mean_reversion": 0.35},
    "TREND_DOWN": {"momentum": 1.3, "breakout": 1.1, "pullback": 1.25, "mean_reversion": 0.35},
    "RANGE": {"momentum": 0.45, "breakout": 0.40, "pullback": 0.65, "mean_reversion": 1.45},
    "HIGH_VOL": {"momentum": 0.9, "breakout": 1.2, "pullback": 0.65, "mean_reversion": 0.45},
    "LOW_VOL": {"momentum": 0.55, "breakout": 0.45, "pullback": 0.8, "mean_reversion": 1.3},
    "SHOCK": {"momentum": 0.45, "breakout": 0.55, "pullback": 0.25, "mean_reversion": 0.20},
    "OPENING_VOLATILITY": {"momentum": 0.85, "breakout": 1.35, "pullback": 0.50, "mean_reversion": 0.30},
}


@dataclass(frozen=True)
class EnsembleSignal:
    p_wait: float
    p_long: float
    p_short: float
    ev_bps: float
    uncertainty: float
    specialist_weights: dict[str, float]

    @property
    def direction(self) -> str:
        return max({"WAIT": self.p_wait, "LONG": self.p_long, "SHORT": self.p_short}, key={"WAIT": self.p_wait, "LONG": self.p_long, "SHORT": self.p_short}.get)

    @property
    def confidence(self) -> float:
        return max(self.p_wait, self.p_long, self.p_short)


class RegimeAwareEnsemble:
    def __init__(self, *, disagreement_wait_boost: float = 0.35) -> None:
        if disagreement_wait_boost < 0:
            raise ValueError("disagreement_wait_boost must be non-negative")
        self.disagreement_wait_boost = disagreement_wait_boost

    def _weight(self, specialist: SpecialistSignal, regime: RegimeState) -> float:
        regime_weight = sum(
            probability * REGIME_WEIGHTS[name].get(specialist.specialist, 0.5)
            for name, probability in regime.probabilities.items()
        )
        certainty = max(0.05, 1.0 - specialist.uncertainty)
        return max(1e-6, regime_weight * certainty)

    def combine(self, signals: Sequence[SpecialistSignal], regime: RegimeState) -> EnsembleSignal:
        if not signals:
            raise ValueError("at least one specialist signal is required")
        raw_weights = np.array([self._weight(signal, regime) for signal in signals], dtype=float)
        weights = raw_weights / raw_weights.sum()
        matrix = np.array([[s.p_wait, s.p_long, s.p_short] for s in signals], dtype=float)
        probs = weights @ matrix

        directional = np.array([s.p_long - s.p_short for s in signals], dtype=float)
        disagreement = float(np.average((directional - np.average(directional, weights=weights)) ** 2, weights=weights))
        wait_boost = min(0.45, self.disagreement_wait_boost * np.sqrt(disagreement))
        probs[0] += wait_boost
        probs[1:] *= max(0.0, 1.0 - wait_boost)
        probs /= probs.sum()
        ev = float(np.average([s.ev_bps for s in signals], weights=weights) * (1.0 - wait_boost))
        entropy = -float(np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)))) / np.log(3.0)
        uncertainty = float(np.clip(0.65 * entropy + 0.35 * min(1.0, disagreement * 8.0), 0.0, 1.0))
        return EnsembleSignal(
            p_wait=float(probs[0]), p_long=float(probs[1]), p_short=float(probs[2]),
            ev_bps=ev, uncertainty=uncertainty,
            specialist_weights={s.specialist: float(w) for s, w in zip(signals, weights)},
        )
