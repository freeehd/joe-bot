"""Calibrated specialist/regime meta-model for Sprint 4."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from strategy.regime import REGIMES, RegimeEngine
from strategy.specialists import Specialist, default_specialists


class SpecialistMetaModel:
    def __init__(self, specialists: list[Specialist] | None = None, regime_engine: RegimeEngine | None = None) -> None:
        self.specialists = specialists or default_specialists()
        self.regime_engine = regime_engine or RegimeEngine()
        self.model = None
        self.feature_columns: list[str] = []

    def meta_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            regime = self.regime_engine.infer_row(row)
            payload: dict[str, float] = {}
            for name in REGIMES:
                payload[f"regime_p_{name.lower()}"] = regime.probabilities[name]
            payload["regime_confidence"] = regime.confidence
            for specialist in self.specialists:
                signal = specialist.evaluate(row, regime)
                prefix = signal.specialist
                payload[f"{prefix}_p_wait"] = signal.p_wait
                payload[f"{prefix}_p_long"] = signal.p_long
                payload[f"{prefix}_p_short"] = signal.p_short
                payload[f"{prefix}_ev_bps"] = signal.ev_bps
                payload[f"{prefix}_uncertainty"] = signal.uncertainty
            # Retain a compact set of context variables so the meta learner can
            # distinguish superficially similar specialist votes.
            for column in (
                "realized_vol_5m", "range_expansion_20", "tod_relative_volume",
                "breadth_up_1m", "minutes_since_open_norm", "relative_strength_spy_5m",
            ):
                value = pd.to_numeric(pd.Series([row.get(column, 0.0)]), errors="coerce").iloc[0]
                payload[f"context_{column}"] = float(value) if np.isfinite(value) else 0.0
            rows.append(payload)
        result = pd.DataFrame(rows, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame, *, method: str = "sigmoid") -> "SpecialistMetaModel":
        train_x = self.meta_features(train)
        cal_x = self.meta_features(calibration)
        self.feature_columns = list(train_x.columns)
        base = LogisticRegression(max_iter=1000, class_weight=None)
        weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
        base.fit(train_x, train["trade_label"].astype(int), sample_weight=weights)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        calibrated.fit(cal_x, calibration["trade_label"].astype(int))
        self.model = calibrated
        return self

    @property
    def classes_(self):
        if self.model is None:
            raise RuntimeError("meta-model is not fitted")
        return self.model.classes_

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("meta-model is not fitted")
        features = self.meta_features(frame)
        return self.model.predict_proba(features[self.feature_columns])
