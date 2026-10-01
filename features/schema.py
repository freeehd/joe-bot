"""Canonical feature schema shared by training and inference."""

FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_5m",
    "ema_distance",
    "vwap_distance",
    "relative_volume",
    "range",
]
