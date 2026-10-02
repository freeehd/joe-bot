"""Victory Sprint 9 shadow/paper campaign metrics and promotion gate.

The campaign layer deliberately scores operational correctness separately from
alpha quality.  A strategy that makes money while duplicating orders, losing
broker position state, or crashing is not promotable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from database.db import AuditStore
from database.models import AuditEvent


@dataclass(frozen=True)
class CampaignThresholds:
    min_shadow_sessions: int = 5
    min_shadow_closed_trades: int = 100
    max_abs_shadow_ev_drift_bps: float = 5.0
    min_shadow_realized_ev_bps: float = 0.0
    min_paper_sessions: int = 10
    min_paper_closed_trades: int = 300
    min_clean_session_rate: float = 1.0
    max_runtime_crashes: int = 0
    max_reconciliation_mismatches: int = 0
    max_duplicate_order_submissions: int = 0
    required_fault_injection_pass_rate: float = 1.0

    def __post_init__(self) -> None:
        if min(self.min_shadow_sessions, self.min_shadow_closed_trades, self.min_paper_sessions, self.min_paper_closed_trades) < 0:
            raise ValueError("campaign minimums must be non-negative")
        if not 0 <= self.min_clean_session_rate <= 1:
            raise ValueError("min_clean_session_rate must be between 0 and 1")
        if not 0 <= self.required_fault_injection_pass_rate <= 1:
            raise ValueError("required_fault_injection_pass_rate must be between 0 and 1")


def _read_events(paths: Sequence[str | Path]) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    for path in paths:
        with AuditStore(path) as store:
            events.extend(list(store.iter_events()))
    return sorted(events, key=lambda event: (event.timestamp, event.event_id or 0))


def _client_order_id(event: AuditEvent) -> str | None:
    payload = event.payload
    intent = payload.get("intent") if isinstance(payload, dict) else None
    broker = payload.get("broker_order") if isinstance(payload, dict) else None
    if isinstance(intent, dict) and intent.get("client_order_id"):
        return str(intent["client_order_id"])
    if isinstance(broker, dict) and broker.get("client_order_id"):
        return str(broker["client_order_id"])
    if isinstance(payload, dict) and payload.get("client_order_id"):
        return str(payload["client_order_id"])
    return None


def _duplicate_submission_count(events: Iterable[AuditEvent]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for event in events:
        if event.event_type not in {"entry_order_submitted", "exit_order_submitted"}:
            continue
        client_id = _client_order_id(event)
        if client_id is None:
            continue
        if client_id in seen:
            duplicates += 1
        seen.add(client_id)
    return duplicates


def _session_cleanliness(events: Sequence[AuditEvent], *, start_types: set[str]) -> dict[str, Any]:
    """Partition an event stream by explicit engine starts and score bad events.

    Session boundaries do not depend on wall-clock date because a process may be
    restarted during the same trading day.  Each explicit start opens a new
    operational session until the next start or the end of the audit stream.
    """
    bad_types = {
        "market_stream_crashed",
        "trade_stream_crashed",
        "shadow_market_stream_crashed",
        "order_reconciliation_mismatch",
    }
    starts = [i for i, event in enumerate(events) if event.event_type in start_types]
    clean = 0
    details: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(events)
        chunk = events[start:end]
        failures: list[str] = []
        for event in chunk:
            if event.event_type in bad_types:
                failures.append(event.event_type)
            if event.event_type == "position_reconciliation" and not bool(event.payload.get("consistent", False)):
                failures.append("position_reconciliation_mismatch")
        is_clean = not failures
        clean += int(is_clean)
        details.append({
            "started_at": chunk[0].timestamp.isoformat(),
            "clean": is_clean,
            "failures": sorted(set(failures)),
        })
    count = len(starts)
    return {
        "sessions": count,
        "clean_sessions": clean,
        "clean_session_rate": clean / count if count else 0.0,
        "session_details": details,
    }


def summarize_shadow_campaign(paths: Sequence[str | Path]) -> dict[str, Any]:
    events = _read_events(paths)
    session = _session_cleanliness(events, start_types={"shadow_mode_started"})
    fills = [event for event in events if event.event_type == "shadow_entry_filled"]
    closes = [event for event in events if event.event_type == "shadow_trade_closed"]
    fill_by_decision = {event.decision_id: event for event in fills if event.decision_id}
    matched = []
    for close in closes:
        if not close.decision_id or close.decision_id not in fill_by_decision:
            continue
        fill = fill_by_decision[close.decision_id]
        expected = fill.payload.get("expected_ev_bps")
        realized = close.payload.get("realized_bps")
        if expected is None or realized is None:
            continue
        matched.append((float(expected), float(realized)))
    expected_values = [pair[0] for pair in matched]
    realized_values = [pair[1] for pair in matched]
    slippage = [float(event.payload["fill_slippage_bps"]) for event in fills if event.payload.get("fill_slippage_bps") is not None]
    latency = [float(event.payload["signal_to_fill_seconds"]) for event in fills if event.payload.get("signal_to_fill_seconds") is not None]
    crashes = sum(1 for event in events if event.event_type.endswith("_crashed"))
    return {
        "mode": "shadow",
        **session,
        "entry_fills": len(fills),
        "closed_trades": len(closes),
        "matched_ev_trades": len(matched),
        "average_expected_ev_bps": mean(expected_values) if expected_values else 0.0,
        "average_realized_ev_bps": mean(realized_values) if realized_values else 0.0,
        "expected_vs_realized_delta_bps": (mean(realized_values) - mean(expected_values)) if matched else 0.0,
        "average_fill_slippage_bps": mean(slippage) if slippage else 0.0,
        "average_signal_to_fill_seconds": mean(latency) if latency else 0.0,
        "runtime_crashes": crashes,
    }


def summarize_paper_campaign(paths: Sequence[str | Path]) -> dict[str, Any]:
    events = _read_events(paths)
    session = _session_cleanliness(events, start_types={"engine_started"})
    submitted_entries = [event for event in events if event.event_type == "entry_order_submitted"]
    submitted_exits = [event for event in events if event.event_type == "exit_order_submitted"]
    updates = [event for event in events if event.event_type == "order_update"]
    filled_client_ids = {
        str(event.payload.get("client_order_id"))
        for event in updates
        if event.payload.get("status") == "FILLED" and event.payload.get("client_order_id")
    }
    exit_client_ids = {_client_order_id(event) for event in submitted_exits}
    closed_trades = sum(1 for client_id in exit_client_ids if client_id and client_id in filled_client_ids)
    position_mismatches = sum(
        1 for event in events
        if event.event_type == "position_reconciliation" and not bool(event.payload.get("consistent", False))
    )
    order_mismatches = sum(1 for event in events if event.event_type == "order_reconciliation_mismatch")
    crashes = sum(1 for event in events if event.event_type.endswith("_crashed"))
    disconnects = sum(1 for event in events if event.event_type in {"broker_disconnected", "market_stream_unstable"})
    duplicates = _duplicate_submission_count(events)
    return {
        "mode": "paper",
        **session,
        "entry_orders_submitted": len(submitted_entries),
        "exit_orders_submitted": len(submitted_exits),
        "filled_order_updates": len(filled_client_ids),
        "closed_trades": closed_trades,
        "position_reconciliation_mismatches": position_mismatches,
        "order_reconciliation_mismatches": order_mismatches,
        "reconciliation_mismatches": position_mismatches + order_mismatches,
        "duplicate_order_submissions": duplicates,
        "runtime_crashes": crashes,
        "disconnect_events": disconnects,
    }


def evaluate_campaign_gate(
    shadow: dict[str, Any],
    paper: dict[str, Any],
    fault_injection: dict[str, Any],
    *,
    thresholds: CampaignThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or CampaignThresholds()
    fault_rate = float(fault_injection.get("pass_rate", 0.0))
    gates = [
        ("shadow_sessions", shadow.get("sessions", 0) >= thresholds.min_shadow_sessions, shadow.get("sessions", 0), f">={thresholds.min_shadow_sessions}"),
        ("shadow_closed_trades", shadow.get("closed_trades", 0) >= thresholds.min_shadow_closed_trades, shadow.get("closed_trades", 0), f">={thresholds.min_shadow_closed_trades}"),
        ("shadow_realized_ev_positive", shadow.get("average_realized_ev_bps", 0.0) > thresholds.min_shadow_realized_ev_bps, shadow.get("average_realized_ev_bps", 0.0), f">{thresholds.min_shadow_realized_ev_bps}"),
        ("shadow_ev_drift_bounded", abs(shadow.get("expected_vs_realized_delta_bps", 0.0)) <= thresholds.max_abs_shadow_ev_drift_bps, abs(shadow.get("expected_vs_realized_delta_bps", 0.0)), f"<={thresholds.max_abs_shadow_ev_drift_bps}"),
        ("paper_sessions", paper.get("sessions", 0) >= thresholds.min_paper_sessions, paper.get("sessions", 0), f">={thresholds.min_paper_sessions}"),
        ("paper_closed_trades", paper.get("closed_trades", 0) >= thresholds.min_paper_closed_trades, paper.get("closed_trades", 0), f">={thresholds.min_paper_closed_trades}"),
        ("clean_session_rate", min(shadow.get("clean_session_rate", 0.0), paper.get("clean_session_rate", 0.0)) >= thresholds.min_clean_session_rate, min(shadow.get("clean_session_rate", 0.0), paper.get("clean_session_rate", 0.0)), f">={thresholds.min_clean_session_rate}"),
        ("no_runtime_crashes", shadow.get("runtime_crashes", 0) + paper.get("runtime_crashes", 0) <= thresholds.max_runtime_crashes, shadow.get("runtime_crashes", 0) + paper.get("runtime_crashes", 0), f"<={thresholds.max_runtime_crashes}"),
        ("no_reconciliation_mismatches", paper.get("reconciliation_mismatches", 0) <= thresholds.max_reconciliation_mismatches, paper.get("reconciliation_mismatches", 0), f"<={thresholds.max_reconciliation_mismatches}"),
        ("no_duplicate_order_submissions", paper.get("duplicate_order_submissions", 0) <= thresholds.max_duplicate_order_submissions, paper.get("duplicate_order_submissions", 0), f"<={thresholds.max_duplicate_order_submissions}"),
        ("fault_injection", fault_rate >= thresholds.required_fault_injection_pass_rate, fault_rate, f">={thresholds.required_fault_injection_pass_rate}"),
    ]
    rows = [
        {"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "mandatory": True}
        for name, passed, value, threshold in gates
    ]
    passed = all(row["passed"] for row in rows)
    return {
        "gate": "SHADOW + PAPER CAMPAIGN",
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "thresholds": asdict(thresholds),
        "gates": rows,
        "shadow": shadow,
        "paper": paper,
        "fault_injection": fault_injection,
        "live_capital_authorized": False,
    }
