"""One-command Victory Sprint 1: acquire, freeze, audit, and study label regimes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.barrier_sweep import run_atr_grid, run_fixed_grid, write_results
from research.build_dataset import build_dataset
from research.dataset_integrity import freeze_dataset
from research.label_tournament import default_configs, run_tournament

UTC = timezone.utc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Joe Bot Victory Sprint 1 end-to-end")
    parser.add_argument("--version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--end", help="UTC ISO timestamp; defaults to current UTC time")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing immutable dataset version")
    args = parser.parse_args()

    if args.years < 2:
        raise ValueError("Victory Sprint 1 requires at least two calendar years")
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else datetime.now(tz=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    else:
        end = end.astimezone(UTC)
    start = end - timedelta(days=365 * args.years + 10)

    if not args.skip_build:
        build_dataset(version=args.version, start=start, end=end, feed=args.feed, root=args.data_root)

    frozen = freeze_dataset(args.version, root=args.data_root)
    if not frozen["all_mandatory_gates_pass"]:
        failed = [g["name"] for g in frozen["gates"] if g["mandatory"] and not g["passed"]]
        raise RuntimeError(f"dataset failed mandatory Victory gates: {failed}")

    experiments = Path(args.data_root) / "experiments" / args.version
    experiments.mkdir(parents=True, exist_ok=True)
    fixed = run_fixed_grid(
        source_version=args.version,
        targets=[0.002, 0.003, 0.004],
        stops=[0.001, 0.0015, 0.002],
        horizons=[5, 10, 15, 20],
        root=args.data_root,
    )
    write_results(fixed, str(experiments / "barrier_fixed.csv"))
    atr = run_atr_grid(
        source_version=args.version,
        target_multipliers=[0.75, 1.0, 1.5],
        stop_multipliers=[0.5, 0.75, 1.0],
        horizons=[5, 10, 15, 20],
        atr_periods=[14],
        root=args.data_root,
    )
    write_results(atr, str(experiments / "barrier_atr.csv"))
    tournament = run_tournament(args.version, default_configs(), root=args.data_root)
    (experiments / "label_tournament.json").write_text(
        json.dumps({
            "source_version": args.version,
            "dataset_fingerprint": frozen["dataset_fingerprint"],
            "test_period_used_for_selection": False,
            "results": tournament,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset_version": args.version,
        "dataset_fingerprint": frozen["dataset_fingerprint"],
        "mandatory_dataset_gates_pass": True,
        "fixed_configs": len(fixed),
        "atr_configs": len(atr),
        "label_tournament_configs": len(tournament),
        "top_label_configs": tournament[:12],
    }
    (experiments / "sprint1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
