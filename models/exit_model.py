"""Chronological continuation-vs-exit model for Sprint 6 research."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

EXIT_FEATURES = [
    "current_return", "mfe_return", "mae_return", "giveback_from_mfe",
    "holding_fraction", "distance_to_target", "distance_to_stop",
    "signal_confidence", "directional_probability", "p_wait",
]


class ExitContinuationModel:
    def __init__(self) -> None:
        self.model = None

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame, *, method: str = "sigmoid") -> "ExitContinuationModel":
        for frame in (train, calibration):
            missing = set(EXIT_FEATURES + ["continue_label"]).difference(frame.columns)
            if missing:
                raise ValueError(f"exit dataset missing columns: {sorted(missing)}")
        base = LogisticRegression(max_iter=1000)
        weights = compute_sample_weight("balanced", train["continue_label"].astype(int))
        base.fit(train[EXIT_FEATURES], train["continue_label"].astype(int), sample_weight=weights)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        calibrated.fit(calibration[EXIT_FEATURES], calibration["continue_label"].astype(int))
        self.model = calibrated
        return self

    def predict_continue_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("exit model is not fitted")
        classes = [int(v) for v in self.model.classes_]
        probs = self.model.predict_proba(frame[EXIT_FEATURES])
        if 1 not in classes:
            return np.zeros(len(frame), dtype=float)
        return probs[:, classes.index(1)]

    def evaluate(self, frame: pd.DataFrame) -> dict:
        y = frame["continue_label"].astype(int).to_numpy()
        p = self.predict_continue_probability(frame)
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
        return {
            "rows": int(len(frame)),
            "continue_rate": float(y.mean()),
            "roc_auc": auc,
            "brier": float(brier_score_loss(y, p)),
        }
