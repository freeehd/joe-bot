"""Transparent Sprint 4 specialist strategies sharing one typed output contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from strategy.regime import RegimeState


def _v(row: Mapping[str, float], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _probabilities(score: float, *, activity: float = 1.0) -> tuple[float, float, float]:
    direction = np.tanh(score)
    strength = min(0.90, abs(direction) * max(0.25, min(activity, 1.5)))
    p_wait = float(np.clip(0.78 - 0.55 * strength, 0.08, 0.92))
    directional = 1.0 - p_wait
    p_long = directional * (0.5 + 0.5 * direction)
    p_short = directional - p_long
    return float(p_wait), float(p_long), float(p_short)


@dataclass(frozen=True)
class SpecialistSignal:
    specialist: str
    p_wait: float
    p_long: float
    p_short: float
    ev_bps: float
    uncertainty: float

    @property
    def direction(self) -> str:
        values = {"WAIT": self.p_wait, "LONG": self.p_long, "SHORT": self.p_short}
        return max(values, key=values.get)

    @property
    def confidence(self) -> float:
        return max(self.p_wait, self.p_long, self.p_short)

    def to_dict(self) -> dict:
        return {
            "specialist": self.specialist,
            "p_wait": self.p_wait,
            "p_long": self.p_long,
            "p_short": self.p_short,
            "ev_bps": self.ev_bps,
            "uncertainty": self.uncertainty,
            "direction": self.direction,
            "confidence": self.confidence,
        }


class Specialist(Protocol):
    name: str
    def evaluate(self, row: Mapping[str, float], regime: RegimeState) -> SpecialistSignal: ...


class _BaseSpecialist:
    name = "base"
    preferred_regimes: tuple[str, ...] = ()

    def _finish(self, raw_score: float, row: Mapping[str, float], regime: RegimeState) -> SpecialistSignal:
        activity = max(0.35, min(1.5, _v(row, "tod_relative_volume", 1.0)))
        p_wait, p_long, p_short = _probabilities(raw_score, activity=activity)
        preferred = sum(regime.probabilities.get(name, 0.0) for name in self.preferred_regimes)
        direction_edge = abs(p_long - p_short)
        uncertainty = float(np.clip(1.0 - (0.55 * direction_edge + 0.45 * preferred), 0.02, 0.98))
        ev_bps = float((max(p_long, p_short) - p_wait) * 18.0 * (0.55 + preferred))
        return SpecialistSignal(self.name, p_wait, p_long, p_short, ev_bps, uncertainty)


class MomentumSpecialist(_BaseSpecialist):
    name = "momentum"
    preferred_regimes = ("TREND_UP", "TREND_DOWN")

    def evaluate(self, row, regime):
        score = (
            45 * _v(row, "return_5m")
            + 24 * _v(row, "return_15m")
            + 18 * _v(row, "ema_5_20_separation")
            + 0.65 * _v(row, "trend_persistence_10")
            + 18 * _v(row, "relative_strength_spy_5m")
        )
        return self._finish(score, row, regime)


class BreakoutSpecialist(_BaseSpecialist):
    name = "breakout"
    preferred_regimes = ("TREND_UP", "TREND_DOWN", "OPENING_VOLATILITY", "HIGH_VOL")

    def evaluate(self, row, regime):
        high = _v(row, "breakout_strength_20_atr")
        low = -_v(row, "bounce_from_session_low_atr") if _v(row, "distance_from_low_20") < 0.002 else 0.0
        volume = np.log1p(max(_v(row, "tod_relative_volume", 1.0), 0.0))
        score = 0.9 * high + 0.45 * low + 0.35 * volume * np.sign(high if high else _v(row, "return_5m"))
        return self._finish(score, row, regime)


class PullbackSpecialist(_BaseSpecialist):
    name = "pullback"
    preferred_regimes = ("TREND_UP", "TREND_DOWN")

    def evaluate(self, row, regime):
        trend = 30 * _v(row, "ema_20_slope_5") + 22 * _v(row, "ema_50_slope_10")
        vwap = _v(row, "vwap_distance")
        pull_from_high = _v(row, "pullback_from_session_high_atr")
        bounce_low = _v(row, "bounce_from_session_low_atr")
        long_setup = max(trend, 0.0) * (0.8 + min(pull_from_high, 2.0)) - 22 * max(vwap, 0.0)
        short_setup = max(-trend, 0.0) * (0.8 + min(bounce_low, 2.0)) - 22 * max(-vwap, 0.0)
        return self._finish(long_setup - short_setup, row, regime)


class MeanReversionSpecialist(_BaseSpecialist):
    name = "mean_reversion"
    preferred_regimes = ("RANGE", "LOW_VOL")

    def evaluate(self, row, regime):
        vwap = _v(row, "vwap_distance")
        ema20 = _v(row, "price_to_ema_20")
        momentum = _v(row, "return_5m")
        trend_penalty = abs(_v(row, "trend_persistence_10"))
        score = -42 * vwap - 26 * ema20 - 18 * momentum
        score *= max(0.15, 1.0 - 0.65 * trend_penalty)
        return self._finish(score, row, regime)


def default_specialists() -> list[Specialist]:
    return [MomentumSpecialist(), BreakoutSpecialist(), PullbackSpecialist(), MeanReversionSpecialist()]
