"""Frozen V0.5 model-bundle loader for live/shadow inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class LiveAlphaModel:
    def __init__(self, model_path: str | Path) -> None:
        bundle = joblib.load(model_path)
        required = {"model", "feature_columns", "class_mapping", "feature_engine"}
        missing = required.difference(bundle)
        if missing:
            raise ValueError(f"alpha bundle missing keys: {sorted(missing)}")
        if bundle["feature_engine"] != "v2":
            raise ValueError("live alpha requires a Feature Engine V2 bundle")
        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.class_mapping = {int(k): str(v) for k, v in bundle["class_mapping"].items()}
        self.metadata = {key: value for key, value in bundle.items() if key != "model"}

    def predict_frame(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"live feature frame missing columns: {sorted(missing)}")
        X = frame[self.feature_columns]
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("live feature frame contains NaN/inf")
        probabilities = self.model.predict_proba(X)
        classes = [int(value) for value in self.model.classes_]
        rows: list[dict[str, Any]] = []
        for probs in probabilities:
            by_class = {class_id: float(prob) for class_id, prob in zip(classes, probs)}
            predicted = max(by_class, key=by_class.get)
            rows.append({
                "p_wait": by_class.get(0, 0.0),
                "p_long": by_class.get(1, 0.0),
                "p_short": by_class.get(2, 0.0),
                "direction": self.class_mapping[predicted],
                "confidence": by_class[predicted],
            })
        return rows
