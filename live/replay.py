"""Replay every persisted audit event associated with one trading decision."""

from __future__ import annotations

import argparse
import json

from database.db import AuditStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Joe Bot paper decision from the append-only audit log")
    parser.add_argument("decision_id")
    parser.add_argument("--audit-db", default="data/paper/audit.sqlite3")
    args = parser.parse_args()

    with AuditStore(args.audit_db) as store:
        print(json.dumps(store.replay(args.decision_id), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
