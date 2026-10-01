"""Expected-value estimation for V0.7 candidate ranking.

The model deliberately separates two sources of information:

1. a structural prior derived from calibrated LONG/WAIT/SHORT probabilities and
   the configured target/stop/cost assumptions; and
2. empirical realized net returns from earlier out-of-sample/paper trades.

Empirical estimates are shrunk toward the structural prior when sample counts
are small. Test-period outcomes must never be passed to ``fit`` before ranking
that same period.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class EVConfig:
    min_samples_per_bucket: int = 10
    shrinkage_samples: float = 40.0
    confidence_edges: tuple[float, ...] = (
        0.0,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.80,
        0.90,
        1.000001,
    )
    wait_return_assumption: float = 0.0

    def __post_init__(self) -> None:
        if self.min_samples_per_bucket < 1:
            raise ValueError("min_samples_per_bucket must be >= 1")
        if self.shrinkage_samples < 0:
            raise ValueError("shrinkage_samples must be >= 0")
        if len(self.confidence_edges) < 2:
            raise ValueError("confidence_edges must contain at least two values")
        if tuple(sorted(self.confidence_edges)) != self.confidence_edges:
            raise ValueError("confidence_edges must be sorted")


@dataclass(frozen=True)
class ExpectedValueEstimate:
    side: Side
    confidence: float
    structural_gross_ev: float
    structural_net_ev: float
    empirical_net_ev: float | None
    blended_net_ev: float
    empirical_samples: int
    empirical_weight: float
    estimated_round_trip_cost: float

    @property
    def net_ev_bps(self) -> float:
        return self.blended_net_ev * 10_000.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["net_ev_bps"] = self.net_ev_bps
        return payload


def estimated_round_trip_cost(config: ExecutionConfig) -> float:
    """Approximate deterministic spread/slippage/fee drag as a return fraction.

    V0.6 applies half the configured full spread on each execution, adverse
    slippage on both entry and exit, and fees on each side. Tick rounding is
    price-dependent and therefore remains in the event-driven backtester rather
    than this ex-ante approximation.
    """

    spread = config.spread_bps / 10_000.0
    slippage = 2.0 * config.slippage_bps / 10_000.0
    fees = 2.0 * config.fee_bps / 10_000.0
    return float(spread + slippage + fees)


def structural_expected_value(
    *,
    side: Side,
    p_wait: float,
    p_long: float,
    p_short: float,
    trade_config: TradeConfig,
    execution_config: ExecutionConfig,
    wait_return_assumption: float = 0.0,
) -> tuple[float, float]:
    probabilities = np.asarray([p_wait, p_long, p_short], dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1")

    if side == "LONG":
        win_probability = p_long
        loss_probability = p_short
    elif side == "SHORT":
        win_probability = p_short
        loss_probability = p_long
    else:
        raise ValueError("side must be LONG or SHORT")

    gross = (
        win_probability * trade_config.target_pct
        - loss_probability * trade_config.stop_pct
        + p_wait * wait_return_assumption
    )
    net = gross - estimated_round_trip_cost(execution_config)
    return float(gross), float(net)


class EmpiricalEVModel:
    """Side/confidence-bucket realized EV model with Bayesian-style shrinkage."""

    def __init__(
        self,
        *,
        trade_config: TradeConfig,
        execution_config: ExecutionConfig,
        config: EVConfig | None = None,
    ) -> None:
        self.trade_config = trade_config
        self.execution_config = execution_config
        self.config = config or EVConfig()
        self._stats: dict[tuple[str, int], dict[str, float | int]] = {}

    def _bucket_index(self, confidence: float) -> int:
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        edges = self.config.confidence_edges
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if low <= confidence < high:
                return index
        return len(edges) - 2

    def fit(self, trades: pd.DataFrame) -> "EmpiricalEVModel":
        required = {"side", "confidence", "net_return"}
        missing = required.difference(trades.columns)
        if missing:
            raise ValueError(f"trades missing columns: {sorted(missing)}")

        self._stats = {}
        if trades.empty:
            return self

        frame = trades.copy()
        frame = frame.loc[frame["side"].isin(["LONG", "SHORT"])].copy()
        frame["confidence_bucket"] = [
            self._bucket_index(float(value)) for value in frame["confidence"]
        ]

        for (side, bucket), group in frame.groupby(["side", "confidence_bucket"]):
            returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
            if returns.empty:
                continue
            self._stats[(str(side), int(bucket))] = {
                "samples": int(len(returns)),
                "mean": float(returns.mean()),
                "std": float(returns.std(ddof=0)),
            }
        return self

    def estimate(
        self,
        *,
        side: Side,
        p_wait: float,
        p_long: float,
        p_short: float,
        confidence: float | None = None,
    ) -> ExpectedValueEstimate:
        confidence = float(
            confidence
            if confidence is not None
            else max(p_long if side == "LONG" else p_short, p_wait)
        )
        gross, structural_net = structural_expected_value(
            side=side,
            p_wait=p_wait,
            p_long=p_long,
            p_short=p_short,
            trade_config=self.trade_config,
            execution_config=self.execution_config,
            wait_return_assumption=self.config.wait_return_assumption,
        )

        bucket = self._bucket_index(confidence)
        stat = self._stats.get((side, bucket))
        empirical: float | None = None
        samples = 0
        weight = 0.0
        blended = structural_net

        if stat is not None:
            samples = int(stat["samples"])
            empirical = float(stat["mean"])
            if samples >= self.config.min_samples_per_bucket:
                denominator = samples + self.config.shrinkage_samples
                weight = float(samples / denominator) if denominator > 0 else 1.0
                blended = weight * empirical + (1.0 - weight) * structural_net

        return ExpectedValueEstimate(
            side=side,
            confidence=confidence,
            structural_gross_ev=gross,
            structural_net_ev=structural_net,
            empirical_net_ev=empirical,
            blended_net_ev=float(blended),
            empirical_samples=samples,
            empirical_weight=weight,
            estimated_round_trip_cost=estimated_round_trip_cost(self.execution_config),
        )

    def stats_frame(self) -> pd.DataFrame:
        rows = []
        edges = self.config.confidence_edges
        for (side, bucket), stats in sorted(self._stats.items()):
            rows.append(
                {
                    "side": side,
                    "bucket": bucket,
                    "low": edges[bucket],
                    "high": min(edges[bucket + 1], 1.0),
                    **stats,
                }
            )
        return pd.DataFrame(rows)
