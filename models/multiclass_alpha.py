"""Inference wrapper for the calibrated V0.4 LONG/WAIT/SHORT alpha model."""

from __future__ import annotations

import joblib
import pandas as pd

from features.schema import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = "data/models/xgboost_multiclass_calibrated.pkl"


class MulticlassAlphaModel:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        self.class_mapping = bundle.get(
            "class_mapping",
            {0: "WAIT", 1: "LONG", 2: "SHORT"},
        )
        self.metadata = {key: value for key, value in bundle.items() if key != "model"}

    def _prepare(self, feature_row) -> pd.DataFrame:
        if isinstance(feature_row, pd.Series):
            feature_row = feature_row.to_frame().T
        return feature_row[self.feature_columns]

    def predict(self, feature_row) -> dict[str, float | str]:
        X = self._prepare(feature_row)
        probabilities = self.model.predict_proba(X)[0]
        classes = [int(value) for value in self.model.classes_]
        by_class = {class_id: float(prob) for class_id, prob in zip(classes, probabilities)}

        wait_probability = by_class.get(0, 0.0)
        long_probability = by_class.get(1, 0.0)
        short_probability = by_class.get(2, 0.0)
        predicted_class = max(by_class, key=by_class.get)

        return {
            "wait_probability": wait_probability,
            "long_probability": long_probability,
            "short_probability": short_probability,
            "direction": self.class_mapping[predicted_class],
            "alpha_probability": by_class[predicted_class],
        }
