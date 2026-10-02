"""Simple, auditable feature-distribution OOD detector for Sprint 5."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class OODResult:
    score: float
    threshold: float
    is_ood: bool
    risk_multiplier: float


class RobustOODDetector:
    """Robust distance detector fitted only on earlier feature distributions.

    Distances are mean squared robust-z values.  The threshold is chosen from a
    separate calibration partition, avoiding test-period threshold selection.
    """

    def __init__(self, *, quantile: float = 0.995, hard_multiplier: float = 1.5) -> None:
        if not 0.9 <= quantile < 1.0:
            raise ValueError("quantile must be in [0.9, 1)")
        if hard_multiplier <= 1.0:
            raise ValueError("hard_multiplier must be > 1")
        self.quantile = quantile
        self.hard_multiplier = hard_multiplier
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.feature_columns: list[str] = []
        self.threshold: float | None = None

    def _clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        if data.isna().any().any():
            raise ValueError("OOD features contain NaN/inf")
        return data

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame, feature_columns: list[str]) -> "RobustOODDetector":
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        self.feature_columns = list(feature_columns)
        train_x = train[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        cal_x = calibration[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        if train_x.isna().any().any() or cal_x.isna().any().any():
            raise ValueError("OOD fit data contains NaN/inf")
        self.scaler.fit(train_x)
        cal_scores = self.score_frame(calibration)
        self.threshold = float(np.quantile(cal_scores, self.quantile))
        return self

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("OOD detector is not fitted")
        transformed = self.scaler.transform(self._clean(frame))
        return np.mean(np.square(np.clip(transformed, -12.0, 12.0)), axis=1)

    def evaluate(self, row: pd.Series | dict) -> OODResult:
        if self.threshold is None:
            raise RuntimeError("OOD detector is not calibrated")
        frame = pd.DataFrame([row]) if isinstance(row, dict) else row.to_frame().T
        score = float(self.score_frame(frame)[0])
        is_ood = score > self.threshold
        if not is_ood:
            multiplier = 1.0
        elif score >= self.threshold * self.hard_multiplier:
            multiplier = 0.0
        else:
            multiplier = float(max(0.1, 1.0 - (score / self.threshold - 1.0) / (self.hard_multiplier - 1.0)))
        return OODResult(score, self.threshold, is_ood, multiplier)
