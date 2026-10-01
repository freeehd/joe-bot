"""Account-level V0.7 portfolio simulation on top of V0.6 trade execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig, simulate_trade
from backtest.metrics import summarize_trades
from market.sectors import sector_for
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2




class CandidateGateProtocol(Protocol):
    def gate_candidates(
        self,
        ranked_candidates: list[dict],
        **context: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    initial_equity: float = 10_000.0
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")


def _max_drawdown(equity_curve: pd.DataFrame) -> tuple[float, float]:
    if equity_curve.empty:
        return 0.0, 0.0
    equity = equity_curve["equity"].astype(float)
    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    minimum = float(drawdown.min())
    dollars = float((equity - peaks).min())
    return minimum, dollars


def _candidate_price(raw: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw bars must use a DatetimeIndex")
    bars = raw.sort_index()
    index = bars.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    matches = bars.loc[index == ts]
    if matches.empty:
        return None
    return float(matches.iloc[-1]["close"])


def run_portfolio_backtest(
    signals: pd.DataFrame,
    *,
    raw_loader: Callable[[str], pd.DataFrame],
    ranker: EVRanker,
    allocator: PortfolioAllocatorV2,
    trade_config: TradeConfig,
    execution_config: ExecutionConfig,
    correlations: pd.DataFrame | None = None,
    config: PortfolioBacktestConfig | None = None,
    symbol_column: str = "training_symbol",
    candidate_gate: CandidateGateProtocol | None = None,
) -> dict:
    """Simulate portfolio selection and realized account equity chronologically.

    The allocator only sees signals and positions known at each timestamp. Trade
    outcomes are generated after selection by V0.6 and are realized only when
    their simulated exit timestamp is reached.
    """

    config = config or PortfolioBacktestConfig()
    required = {symbol_column, "p_wait", "p_long", "p_short", "direction", "confidence"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError("signals must use a DatetimeIndex")

    equity = float(config.initial_equity)
    open_positions: list[dict] = []
    closed_records: list[dict] = []
    equity_events = [
        {
            "timestamp": signals.index.min() if len(signals) else pd.Timestamp.now(tz="UTC"),
            "equity": equity,
            "event": "START",
        }
    ]
    raw_cache: dict[str, pd.DataFrame] = {}
    allocation_snapshots: list[dict] = []
    gate_snapshots: list[dict] = []

    def raw_for(symbol: str) -> pd.DataFrame:
        if symbol not in raw_cache:
            raw_cache[symbol] = raw_loader(symbol)
        return raw_cache[symbol]

    def realize_through(timestamp: pd.Timestamp) -> None:
        nonlocal equity, open_positions
        remaining = []
        for position in sorted(open_positions, key=lambda item: item["exit_time"]):
            if position["exit_time"] <= timestamp:
                equity += float(position["pnl_dollars"])
                record = position.copy()
                record["equity_after_exit"] = equity
                closed_records.append(record)
                equity_events.append(
                    {
                        "timestamp": position["exit_time"],
                        "equity": equity,
                        "event": f"EXIT_{position['symbol']}",
                    }
                )
            else:
                remaining.append(position)
        open_positions = remaining

    for timestamp, group in signals.sort_index().groupby(level=0, sort=True):
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        realize_through(ts)

        candidates: list[dict] = []
        for _, row in group.iterrows():
            direction = str(row["direction"])
            confidence = float(row["confidence"])
            if direction not in {"LONG", "SHORT"} or confidence < config.min_confidence:
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
                    "confidence": confidence,
                    "p_wait": float(row["p_wait"]),
                    "p_long": float(row["p_long"]),
                    "p_short": float(row["p_short"]),
                    "price": price,
                    "sector": sector_for(symbol),
                    "signal_time": ts,
                }
            )
            candidates.append(candidate)

        if not candidates:
            continue

        current_for_allocator = [
            {
                "symbol": position["symbol"],
                "direction": position["direction"],
                "sector": position["sector"],
                "allocation": position["allocation"],
                "risk_dollars": position["risk_dollars"],
            }
            for position in open_positions
        ]
        ranked = ranker.rank(
            candidates,
            correlations=correlations,
            current_positions=current_for_allocator,
            min_net_ev_bps=allocator.constraints.min_net_ev_bps,
        )
        if candidate_gate is not None and ranked:
            gate_result = candidate_gate.gate_candidates(
                ranked,
                timestamp=ts,
                account_equity=equity,
                current_positions=current_for_allocator,
                correlations=correlations,
            )
            ranked = [item.copy() for item in gate_result.get("approved", [])]
            gate_snapshots.append(
                {
                    "timestamp": ts.isoformat(),
                    "quant_candidates": len(candidates),
                    "quant_positive_ev": len(gate_result.get("approved", [])) + len(gate_result.get("vetoed", [])),
                    "laya_approved": len(gate_result.get("approved", [])),
                    "laya_vetoed": len(gate_result.get("vetoed", [])),
                    "decisions": gate_result.get("decisions", []),
                    "error": gate_result.get("error"),
                }
            )
        allocation = allocator.allocate(
            ranked,
            account_equity=equity,
            correlations=correlations,
            current_positions=current_for_allocator,
        )
        allocation_snapshots.append(
            {
                "timestamp": ts.isoformat(),
                "equity": equity,
                "candidates": len(candidates),
                "positive_ev_candidates": len(ranked),
                "selected": allocation["selected_count"],
                "capital_deployed": allocation["capital_deployed"],
                "portfolio_risk_dollars": allocation["portfolio_risk_dollars"],
            }
        )

        for selected in allocation["positions"]:
            symbol = str(selected["symbol"])
            result = simulate_trade(
                raw_for(symbol),
                symbol=symbol,
                signal_time=ts,
                side=str(selected["direction"]),  # type: ignore[arg-type]
                confidence=float(selected["confidence"]),
                p_wait=float(selected["p_wait"]),
                p_long=float(selected["p_long"]),
                p_short=float(selected["p_short"]),
                trade_config=trade_config,
                execution_config=execution_config,
            )
            if result is None:
                continue

            quantity = int(selected["quantity"])
            actual_notional = float(quantity * result.entry_price)
            actual_risk = float(actual_notional * selected["stop_pct"])
            pnl_dollars = float(actual_notional * result.net_return)
            record = {
                **result.to_dict(),
                "signal_time": result.signal_time,
                "entry_time": result.entry_time,
                "exit_time": result.exit_time,
                "sector": selected["sector"],
                "quantity": quantity,
                "allocation": actual_notional,
                "risk_dollars": actual_risk,
                "pnl_dollars": pnl_dollars,
                "net_ev_bps_at_entry": float(selected["net_ev_bps"]),
                "opportunity_score": float(selected["opportunity_score"]),
                "rank": int(selected["rank"]),
                "max_directional_correlation": float(selected["max_directional_correlation"]),
            }
            open_positions.append(record)

    # Realize all remaining positions at their already-simulated exit times.
    if open_positions:
        realize_through(max(position["exit_time"] for position in open_positions))

    trades = pd.DataFrame(closed_records)
    equity_curve = pd.DataFrame(equity_events).sort_values("timestamp").reset_index(drop=True)
    max_dd, max_dd_dollars = _max_drawdown(equity_curve)
    trade_summary = summarize_trades(trades if not trades.empty else pd.DataFrame())

    gross_profit = float(trades.loc[trades["pnl_dollars"] > 0, "pnl_dollars"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["pnl_dollars"] < 0, "pnl_dollars"].sum()) if not trades.empty else 0.0
    profit_factor_dollars = (
        float("inf") if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else 0.0)
    )

    return {
        "config": asdict(config),
        "initial_equity": config.initial_equity,
        "ending_equity": equity,
        "net_pnl_dollars": equity - config.initial_equity,
        "total_return": equity / config.initial_equity - 1.0,
        "max_drawdown": max_dd,
        "max_drawdown_dollars": max_dd_dollars,
        "profit_factor_dollars": profit_factor_dollars,
        "trade_metrics": trade_summary,
        "trade_records": trades,
        "equity_curve": equity_curve,
        "allocation_snapshots": allocation_snapshots,
        "gate_snapshots": gate_snapshots,
    }
