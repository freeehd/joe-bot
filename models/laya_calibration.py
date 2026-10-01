"""Post-hoc probability calibration utilities for V0.8 Laya decisions.

Laya checkpoints carry their own temperatures, but the trading specialist must
still be measured and calibrated on Joe Bot's held-out trading distribution.
These helpers fit a simple additional temperature directly from probabilities,
which keeps the evaluation path independent of the Laya training runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from models.laya_engine import LayaDecision


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D matrix")
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    rows = np.arange(len(labels))
    chosen = np.clip(probabilities[rows, labels], 1e-12, 1.0)
    return float(-np.log(chosen).mean())


def fit_temperature(probabilities: np.ndarray, labels: Iterable[int]) -> float:
    """Fit one scalar temperature by deterministic coarse-to-fine grid search."""

    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if probs.ndim != 2 or len(probs) != len(y) or len(y) == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("labels outside probability columns")

    best_t = 1.0
    best_loss = float("inf")
    for low, high, count in ((0.25, 4.0, 151), (None, None, 101)):
        if low is None:
            low = max(0.05, best_t - 0.25)
            high = best_t + 0.25
        for temperature in np.linspace(low, high, count):
            loss = _negative_log_likelihood(temperature_scale(probs, float(temperature)), y)
            if loss < best_loss:
                best_loss = loss
                best_t = float(temperature)
    return best_t


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Iterable[int],
    *,
    bins: int = 10,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if len(y) == 0:
        return 0.0
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (predicted == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


@dataclass(frozen=True)
class LayaCalibration:
    action_temperature: float = 1.0
    quality_temperature: float = 1.0
    risk_temperature: float = 1.0
    fitted_on: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LayaCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def calibrate(self, decision: LayaDecision) -> LayaDecision:
        action_labels = ["LONG", "WAIT", "SHORT"]
        action_raw = np.array([[decision.action_probabilities.get(label, 0.0) for label in action_labels]])
        action_scaled = temperature_scale(action_raw, self.action_temperature)[0]
        action_probs = dict(zip(action_labels, map(float, action_scaled)))
        action = max(action_probs, key=action_probs.get)

        quality_labels = ["POOR", "FAIR", "GOOD", "EXCELLENT"]
        quality_raw = np.array([[decision.quality_probabilities.get(label, 0.0) for label in quality_labels]])
        if quality_raw.sum() <= 0:
            quality_probs = decision.quality_probabilities.copy()
            quality = decision.quality
            quality_confidence = decision.quality_answer_confidence
        else:
            quality_scaled = temperature_scale(quality_raw, self.quality_temperature)[0]
            quality_probs = dict(zip(quality_labels, map(float, quality_scaled)))
            quality = max(quality_probs, key=quality_probs.get)
            quality_confidence = float(max(quality_probs.values()))

        risk_raw = np.array([[1.0 - decision.risk_concern_probability, decision.risk_concern_probability]])
        risk_scaled = temperature_scale(risk_raw, self.risk_temperature)[0]

        return LayaDecision(
            action=action,
            action_probabilities=action_probs,
            action_answer_confidence=float(max(action_probs.values())),
            quality=quality,
            quality_probabilities=quality_probs,
            quality_answer_confidence=quality_confidence,
            risk_concern_probability=float(risk_scaled[1]),
            model=decision.model,
        )
