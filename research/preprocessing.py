"""Shared feature/label preprocessing for immutable research datasets."""

from __future__ import annotations

import pandas as pd

from features.v2 import (
    FEATURE_COLUMNS_V2,
    MARKET_CONTEXT_FEATURES,
    add_market_context,
    create_features_v2,
)
from labels.triple_barrier import BarrierConfig, create_multiclass_labels

SINGLE_SYMBOL_FEATURES_V2 = [
    feature for feature in FEATURE_COLUMNS_V2 if feature not in MARKET_CONTEXT_FEATURES
]


def build_symbol_frame_v2(
    raw_df: pd.DataFrame,
    *,
    symbol: str,
    barrier_config: BarrierConfig,
) -> pd.DataFrame:
    """Build causal V2 single-symbol features + labels, before market context."""

    frame = raw_df.copy().sort_index()
    if "symbol" in frame.index.names:
        frame = frame.reset_index(level="symbol", drop=True)

    frame = create_features_v2(frame)
    frame = create_multiclass_labels(frame, barrier_config, respect_sessions=True)
    frame["training_symbol"] = symbol
    required = SINGLE_SYMBOL_FEATURES_V2 + ["trade_label"]
    frame = frame[frame["label_valid"]].dropna(subset=required)
    frame["trade_label"] = frame["trade_label"].astype(int)
    return frame


def add_context_and_finalize_v2(
    symbol_frames: list[pd.DataFrame],
    *,
    training_symbols: list[str],
) -> pd.DataFrame:
    """Add timestamp-aligned market context and retain model-training symbols."""

    if not symbol_frames:
        raise ValueError("No symbol frames supplied")
    combined = pd.concat(symbol_frames).sort_index()
    combined = add_market_context(combined)
    combined = combined.loc[combined["training_symbol"].isin(training_symbols)].copy()
    combined = combined.dropna(subset=FEATURE_COLUMNS_V2 + ["trade_label"])
    combined["trade_label"] = combined["trade_label"].astype(int)
    return combined.sort_index()
