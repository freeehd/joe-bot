"""Historical V0.8 Laya specialist dataset construction.

Training labels are derived from realized V0.6 execution outcomes, never from
the quantitative model's predicted direction. The generated records preserve
the quant proposal in the state so Laya can learn when it should veto it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig, TradeResult, simulate_trade
from market.sectors import sector_for
from models.laya_engine import LAYA_TRADING_QUESTIONS
from strategy.laya_gate import build_laya_state
from strategy.portfolio_allocator import EVRanker


@dataclass(frozen=True)
class LayaLabelConfig:
    min_best_return_bps: float = 0.0
    min_direction_margin_bps: float = 1.0
    good_fraction_of_target: float = 0.35
    excellent_fraction_of_target: float = 0.75

    def __post_init__(self) -> None:
        if self.min_direction_margin_bps < 0:
            raise ValueError("min_direction_margin_bps must be non-negative")
        if not 0 <= self.good_fraction_of_target <= self.excellent_fraction_of_target:
            raise ValueError("quality thresholds must be ordered and non-negative")


@dataclass(frozen=True)
class LayaGroundTruth:
    action: str
    quality: str
    risk_concern: bool
    long_net_return: float | None
    short_net_return: float | None
    best_net_return: float
    winning_exit_reason: str | None

    def answers(self) -> dict:
        return {
            "action": self.action,
            "quality": self.quality,
            "risk_concern": self.risk_concern,
        }


def _utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _candidate_price(raw: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    bars = raw.sort_index()
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("raw bars must use a DatetimeIndex")
    index = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
    matches = bars.loc[index == _utc(timestamp)]
    if matches.empty:
        return None
    return float(matches.iloc[-1]["close"])


def derive_laya_ground_truth(
    raw_bars: pd.DataFrame,
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    proposed_direction: str,
    probabilities: dict[str, float],
    confidence: float,
    trade_config: TradeConfig,
    execution_config: ExecutionConfig,
    label_config: LayaLabelConfig | None = None,
) -> tuple[LayaGroundTruth, dict[str, TradeResult | None]]:
    """Label the historical state using realized executable LONG/SHORT outcomes."""

    cfg = label_config or LayaLabelConfig()
    common = {
        "symbol": symbol,
        "signal_time": _utc(timestamp),
        "confidence": float(confidence),
        "p_wait": float(probabilities["p_wait"]),
        "p_long": float(probabilities["p_long"]),
        "p_short": float(probabilities["p_short"]),
        "trade_config": trade_config,
        "execution_config": execution_config,
    }
    long_result = simulate_trade(raw_bars, side="LONG", **common)
    short_result = simulate_trade(raw_bars, side="SHORT", **common)
    outcomes = {"LONG": long_result, "SHORT": short_result}

    long_return = None if long_result is None else float(long_result.net_return)
    short_return = None if short_result is None else float(short_result.net_return)
    comparable = {
        side: result
        for side, result in outcomes.items()
        if result is not None
    }
    if not comparable:
        truth = LayaGroundTruth(
            action="WAIT",
            quality="POOR",
            risk_concern=True,
            long_net_return=long_return,
            short_net_return=short_return,
            best_net_return=0.0,
            winning_exit_reason=None,
        )
        return truth, outcomes

    ordered = sorted(comparable.items(), key=lambda item: float(item[1].net_return), reverse=True)
    best_side, best_result = ordered[0]
    best_return = float(best_result.net_return)
    second_return = float(ordered[1][1].net_return) if len(ordered) > 1 else float("-inf")
    margin_bps = (best_return - second_return) * 10_000.0

    if best_return * 10_000.0 <= cfg.min_best_return_bps or margin_bps < cfg.min_direction_margin_bps:
        action = "WAIT"
        quality = "POOR"
        winning_exit_reason = None
    else:
        action = best_side
        fraction = best_return / trade_config.target_pct if trade_config.target_pct > 0 else 0.0
        if fraction >= cfg.excellent_fraction_of_target:
            quality = "EXCELLENT"
        elif fraction >= cfg.good_fraction_of_target:
            quality = "GOOD"
        else:
            quality = "FAIR"
        winning_exit_reason = best_result.exit_reason

    risk_concern = action == "WAIT" or action != proposed_direction
    truth = LayaGroundTruth(
        action=action,
        quality=quality,
        risk_concern=risk_concern,
        long_net_return=long_return,
        short_net_return=short_return,
        best_net_return=max(best_return, 0.0),
        winning_exit_reason=winning_exit_reason,
    )
    return truth, outcomes


def build_laya_examples(
    signals: pd.DataFrame,
    *,
    raw_loader: Callable[[str], pd.DataFrame],
    ranker: EVRanker,
    trade_config: TradeConfig,
    execution_config: ExecutionConfig,
    correlations: pd.DataFrame | None = None,
    top_candidates: int = 5,
    label_config: LayaLabelConfig | None = None,
    symbol_column: str = "training_symbol",
    source_window: int | None = None,
) -> list[dict]:
    """Convert a held-out signal frame into supervised Laya examples."""

    if top_candidates < 1:
        raise ValueError("top_candidates must be >= 1")
    required = {symbol_column, "p_wait", "p_long", "p_short", "direction", "confidence"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError("signals must use a DatetimeIndex")

    raw_cache: dict[str, pd.DataFrame] = {}

    def raw_for(symbol: str) -> pd.DataFrame:
        if symbol not in raw_cache:
            raw_cache[symbol] = raw_loader(symbol)
        return raw_cache[symbol]

    examples: list[dict] = []
    for timestamp, group in signals.sort_index().groupby(level=0, sort=True):
        ts = _utc(pd.Timestamp(timestamp))
        candidates: list[dict] = []
        for _, row in group.iterrows():
            direction = str(row["direction"])
            if direction not in {"LONG", "SHORT"}:
                continue
            symbol = str(row[symbol_column])
            price = _candidate_price(raw_for(symbol), ts)
            if price is None or price <= 0:
                continue
            candidate = row.to_dict()
            candidate.update(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": float(row["confidence"]),
                    "p_wait": float(row["p_wait"]),
                    "p_long": float(row["p_long"]),
                    "p_short": float(row["p_short"]),
                    "price": price,
                    "sector": sector_for(symbol),
                    "signal_time": ts,
                }
            )
            candidates.append(candidate)

        ranked = ranker.rank(candidates, correlations=correlations, min_net_ev_bps=0.0)
        for candidate in ranked[:top_candidates]:
            symbol = str(candidate["symbol"])
            truth, outcomes = derive_laya_ground_truth(
                raw_for(symbol),
                symbol=symbol,
                timestamp=ts,
                proposed_direction=str(candidate["direction"]),
                probabilities={
                    "p_wait": float(candidate["p_wait"]),
                    "p_long": float(candidate["p_long"]),
                    "p_short": float(candidate["p_short"]),
                },
                confidence=float(candidate["confidence"]),
                trade_config=trade_config,
                execution_config=execution_config,
                label_config=label_config,
            )
            state = build_laya_state(candidate, timestamp=ts, account_equity=None, current_positions=[])
            record = {
                "state": state,
                "questions": LAYA_TRADING_QUESTIONS,
                "answers": truth.answers(),
                "metadata": {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "quant_direction": candidate["direction"],
                    "quant_net_ev_bps": float(candidate.get("net_ev_bps", 0.0)),
                    "quant_rank": int(candidate.get("rank", 0)),
                    "source_window": source_window,
                    "ground_truth": asdict(truth),
                    "quant_trade_net_return": (
                        None
                        if outcomes.get(str(candidate["direction"])) is None
                        else float(outcomes[str(candidate["direction"])].net_return)
                    ),
                },
            }
            examples.append(record)
    return examples


def split_chronologically(
    examples: list[dict],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[dict]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("validation_fraction leaves no held-out test data")
    ordered = sorted(examples, key=lambda item: item["metadata"]["timestamp"])
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def write_jsonl(records: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def write_laya_dataset(
    examples: list[dict],
    output_dir: str | Path,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    manifest_extra: dict | None = None,
) -> dict:
    root = Path(output_dir)
    splits = split_chronologically(
        examples,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    for split, records in splits.items():
        write_jsonl(records, root / f"{split}.jsonl")

    manifest = {
        "records": len(examples),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "label_source": "V0.6 realized LONG/SHORT execution outcomes",
        "questions": LAYA_TRADING_QUESTIONS,
        **(manifest_extra or {}),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
