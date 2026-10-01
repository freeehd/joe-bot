"""Convert calibrated class probabilities into explicit research signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "WAIT", 1: "LONG", 2: "SHORT"}


def probability_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    classes=(0, 1, 2),
) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(frame), len(classes)):
        raise ValueError("probability matrix shape does not match frame/classes")

    by_class = {int(class_id): probabilities[:, i] for i, class_id in enumerate(classes)}
    for required in (0, 1, 2):
        if required not in by_class:
            raise ValueError(f"missing probability class {required}")

    signals = frame.copy()
    signals["p_wait"] = by_class[0]
    signals["p_long"] = by_class[1]
    signals["p_short"] = by_class[2]
    class_matrix = np.column_stack([signals["p_wait"], signals["p_long"], signals["p_short"]])
    predicted = class_matrix.argmax(axis=1)
    signals["direction"] = [CLASS_NAMES[int(value)] for value in predicted]
    signals["confidence"] = class_matrix.max(axis=1)
    signals["directional_edge"] = np.maximum(signals["p_long"], signals["p_short"]) - signals["p_wait"]
    return signals
