"""Operational SLO evaluation for Joe Bot services."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import quantiles


@dataclass(frozen=True)
class SLOThresholds:
    min_availability: float = 0.999
    max_p95_latency_ms: float = 1000.0
    max_error_rate: float = 0.001


def _p95(values: list[float]) -> float:
    if not values:
        return float("inf")
    if len(values) < 20:
        return sorted(values)[max(0, int(round(0.95 * (len(values) - 1))))]
    return quantiles(values, n=100, method="inclusive")[94]


def evaluate_slo(*, total_checks: int, successful_checks: int, errors: int, latencies_ms: list[float], thresholds: SLOThresholds | None = None) -> dict:
    thresholds = thresholds or SLOThresholds()
    availability = successful_checks / total_checks if total_checks else 0.0
    error_rate = errors / total_checks if total_checks else 1.0
    p95 = _p95(latencies_ms)
    gates = {
        "availability": availability >= thresholds.min_availability,
        "p95_latency": p95 <= thresholds.max_p95_latency_ms,
        "error_rate": error_rate <= thresholds.max_error_rate,
    }
    return {
        "passed": all(gates.values()),
        "availability": availability,
        "error_rate": error_rate,
        "p95_latency_ms": p95,
        "gates": gates,
    }
