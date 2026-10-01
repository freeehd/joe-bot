"""Dataset diagnostics used before a model is allowed to train."""

from __future__ import annotations

from collections import Counter
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
LABEL_NAMES = {0: "WAIT", 1: "LONG", 2: "SHORT"}


def class_distribution(df: pd.DataFrame) -> dict:
    counts = Counter(int(value) for value in df["trade_label"].dropna())
    total = sum(counts.values())
    return {
        LABEL_NAMES[class_id]: {
            "count": int(counts.get(class_id, 0)),
            "percent": round(100.0 * counts.get(class_id, 0) / total, 4) if total else 0.0,
        }
        for class_id in (0, 1, 2)
    }


def outcome_distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def hourly_label_distribution(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty:
        return {}
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert(NY)
    report = df[["trade_label"]].copy()
    report["hour_et"] = local.strftime("%H:%M").str.slice(0, 2)

    result: dict[str, dict] = {}
    for hour, group in report.groupby("hour_et"):
        result[str(hour)] = class_distribution(group)
    return result


def per_symbol_distribution(df: pd.DataFrame) -> dict[str, dict]:
    if "training_symbol" not in df.columns:
        return {}
    return {
        str(symbol): class_distribution(group)
        for symbol, group in df.groupby("training_symbol")
    }
