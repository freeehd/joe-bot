"""Regime/specialist diagnostics for Victory Sprint 4."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from strategy.ensemble import RegimeAwareEnsemble
from strategy.regime import RegimeEngine
from strategy.specialists import Specialist, default_specialists


def attach_regimes(frame: pd.DataFrame, engine: RegimeEngine | None = None) -> pd.DataFrame:
    engine = engine or RegimeEngine()
    result = frame.copy()
    regime = engine.transform(result)
    for column in regime.columns:
        result[column] = regime[column].to_numpy()
    return result


def specialist_probability_frame(
    frame: pd.DataFrame,
    *,
    specialists: Iterable[Specialist] | None = None,
    regime_engine: RegimeEngine | None = None,
    ensemble: RegimeAwareEnsemble | None = None,
) -> pd.DataFrame:
    specialists = list(specialists or default_specialists())
    regime_engine = regime_engine or RegimeEngine()
    ensemble = ensemble or RegimeAwareEnsemble()
    rows = []
    for _, row in frame.iterrows():
        regime = regime_engine.infer_row(row)
        signals = [specialist.evaluate(row, regime) for specialist in specialists]
        combined = ensemble.combine(signals, regime)
        record = {
            "p_wait": combined.p_wait,
            "p_long": combined.p_long,
            "p_short": combined.p_short,
            "confidence": combined.confidence,
            "direction": combined.direction,
            "ensemble_ev_bps": combined.ev_bps,
            "ensemble_uncertainty": combined.uncertainty,
            "regime": regime.label,
            "regime_confidence": regime.confidence,
        }
        for signal in signals:
            record[f"{signal.specialist}_p_wait"] = signal.p_wait
            record[f"{signal.specialist}_p_long"] = signal.p_long
            record[f"{signal.specialist}_p_short"] = signal.p_short
            record[f"{signal.specialist}_ev_bps"] = signal.ev_bps
            record[f"{signal.specialist}_uncertainty"] = signal.uncertainty
        rows.append(record)
    result = frame.copy()
    extra = pd.DataFrame(rows, index=frame.index)
    for column in extra.columns:
        result[column] = extra[column]
    return result


def trade_performance_by_regime(trades: pd.DataFrame) -> list[dict]:
    if trades.empty or "regime" not in trades.columns:
        return []
    rows = []
    for regime, group in trades.groupby("regime"):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append({
            "regime": str(regime),
            "trades": int(len(returns)),
            "expectancy_bps": float(returns.mean() * 10_000.0),
            "win_rate": float((returns > 0).mean()),
            "net_return_sum": float(returns.sum()),
        })
    return sorted(rows, key=lambda row: row["trades"], reverse=True)
