"""Summaries for V0.95 shadow-production audit databases."""
from __future__ import annotations

import argparse
import json
from statistics import mean

from database.db import AuditStore


def summarize_shadow(store: AuditStore) -> dict:
    events = list(store.iter_events())
    candidates = [e for e in events if e.event_type == "shadow_candidate"]
    risk = [e for e in events if e.event_type == "risk_decision"]
    fills = [e for e in events if e.event_type == "shadow_entry_filled"]
    closes = [e for e in events if e.event_type == "shadow_trade_closed"]
    checkpoints = [e for e in events if e.event_type == "shadow_checkpoint"]

    approved = sum(1 for e in risk if bool(e.payload.get("risk", {}).get("approved")))
    expected = [float(e.payload["expected_ev_bps"]) for e in fills if e.payload.get("expected_ev_bps") is not None]
    realized = [float(e.payload["realized_bps"]) for e in closes if e.payload.get("realized_bps") is not None]
    slippage = [float(e.payload["fill_slippage_bps"]) for e in fills if e.payload.get("fill_slippage_bps") is not None]
    latency = [float(e.payload["signal_to_fill_seconds"]) for e in fills if e.payload.get("signal_to_fill_seconds") is not None]
    wins = sum(1 for value in realized if value > 0)
    crashes = [e for e in events if e.event_type.endswith("_crashed")]

    return {
        "mode": "shadow",
        "external_orders": False,
        "candidates": len(candidates),
        "risk_approved": approved,
        "entry_fills": len(fills),
        "closed_trades": len(closes),
        "average_expected_ev_bps": mean(expected) if expected else 0.0,
        "average_realized_bps": mean(realized) if realized else 0.0,
        "expected_vs_realized_delta_bps": (mean(realized) - mean(expected)) if realized and expected else 0.0,
        "average_fill_slippage_bps": mean(slippage) if slippage else 0.0,
        "average_signal_to_fill_seconds": mean(latency) if latency else 0.0,
        "runtime_crashes": len(crashes),
        "win_rate": wins / len(realized) if realized else 0.0,
        "positive_realized_ev": bool(realized and mean(realized) > 0),
        "checkpoints": len(checkpoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Joe Bot V0.95 shadow audit database")
    parser.add_argument("--audit-db", default="data/shadow/audit.sqlite3")
    args = parser.parse_args()
    with AuditStore(args.audit_db) as store:
        print(json.dumps(summarize_shadow(store), indent=2))


if __name__ == "__main__":
    main()
