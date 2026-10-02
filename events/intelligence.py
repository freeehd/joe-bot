"""Validated structured event/news intelligence for GOLD 24.

No class in this module exposes broker/order methods.  External NLP/LLM systems may
produce dictionaries, but they must pass this schema before the result can enter
alpha/meta/risk features.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Protocol

UTC = timezone.utc


class EventDirection(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class EventImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventHorizon(str, Enum):
    MINUTES = "minutes"
    INTRADAY = "intraday"
    MULTIDAY = "multiday"


_IMPACT_WEIGHT = {EventImpact.LOW: 0.35, EventImpact.MEDIUM: 0.65, EventImpact.HIGH: 1.0}
_DIRECTION = {EventDirection.NEGATIVE: -1.0, EventDirection.NEUTRAL: 0.0, EventDirection.POSITIVE: 1.0}


@dataclass(frozen=True)
class StructuredEvent:
    event_type: str
    direction: EventDirection
    surprise: float
    impact: EventImpact
    expected_horizon: EventHorizon
    credibility: float
    symbols: tuple[str, ...]
    published_at: datetime
    source_id: str
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.event_type.strip() or not self.source_id.strip():
            raise ValueError("event_type and source_id are required")
        if not 0.0 <= self.surprise <= 1.0 or not 0.0 <= self.credibility <= 1.0:
            raise ValueError("surprise and credibility must be in [0,1]")
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        object.__setattr__(self, "symbols", tuple(dict.fromkeys(s.upper() for s in self.symbols)))
        ts = self.published_at
        object.__setattr__(self, "published_at", ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC))

    @property
    def signed_score(self) -> float:
        return _DIRECTION[self.direction] * self.surprise * self.credibility * _IMPACT_WEIGHT[self.impact]

    def to_features(self) -> dict[str, Any]:
        return {
            "event_score": self.signed_score,
            "event_surprise": self.surprise,
            "event_credibility": self.credibility,
            "event_high_impact": float(self.impact == EventImpact.HIGH),
            "event_positive": float(self.direction == EventDirection.POSITIVE),
            "event_negative": float(self.direction == EventDirection.NEGATIVE),
            "event_intraday": float(self.expected_horizon in {EventHorizon.MINUTES, EventHorizon.INTRADAY}),
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["direction"] = self.direction.value
        value["impact"] = self.impact.value
        value["expected_horizon"] = self.expected_horizon.value
        value["published_at"] = self.published_at.isoformat()
        value["signed_score"] = self.signed_score
        return value


class EventExtractor(Protocol):
    def extract(self, payload: Any) -> StructuredEvent: ...


def validate_event_payload(payload: dict[str, Any]) -> StructuredEvent:
    """Turn untrusted NLP output into a strongly typed, bounded event feature."""
    allowed = {
        "event_type", "direction", "surprise", "impact", "expected_horizon",
        "credibility", "symbols", "published_at", "source_id", "summary",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"unknown event fields: {sorted(unknown)}")
    published = payload["published_at"]
    if not isinstance(published, datetime):
        published = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
    return StructuredEvent(
        event_type=str(payload["event_type"]),
        direction=EventDirection(str(payload["direction"]).lower()),
        surprise=float(payload["surprise"]),
        impact=EventImpact(str(payload["impact"]).lower()),
        expected_horizon=EventHorizon(str(payload["expected_horizon"]).lower()),
        credibility=float(payload["credibility"]),
        symbols=tuple(str(x) for x in payload["symbols"]),
        published_at=published,
        source_id=str(payload["source_id"]),
        summary=str(payload.get("summary", "")),
    )


def aggregate_symbol_events(events: Iterable[StructuredEvent], *, now: datetime, max_age_minutes: int = 390) -> dict[str, dict[str, float]]:
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    totals: dict[str, list[float]] = {}
    for event in events:
        age = max(0.0, (now - event.published_at).total_seconds() / 60.0)
        if age > max_age_minutes:
            continue
        decay = max(0.0, 1.0 - age / max_age_minutes)
        for symbol in event.symbols:
            totals.setdefault(symbol, []).append(event.signed_score * decay)
    return {
        symbol: {
            "event_score": sum(values),
            "event_count": float(len(values)),
            "event_abs_score": sum(abs(v) for v in values),
        }
        for symbol, values in totals.items()
    }
