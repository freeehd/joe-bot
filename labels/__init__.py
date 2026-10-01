"""Ground-truth labeling utilities for the trading research pipeline."""

from .triple_barrier import (
    BarrierConfig,
    BarrierOutcome,
    TradeLabel,
    create_multiclass_labels,
    evaluate_barriers,
)

__all__ = [
    "BarrierConfig",
    "BarrierOutcome",
    "TradeLabel",
    "create_multiclass_labels",
    "evaluate_barriers",
]
