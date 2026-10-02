"""Dataset integrity, freezing, and quality-gate tooling."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.artifact_registry import canonical_json_bytes, sha256_bytes, sha256_file
from research.storage import ParquetDataLake

UTC = timezone.utc


def _resolved_files(manifest: dict, root: Path) -> list[Path]:
    result: list[Path] = []
    for group in ("raw_partitions", "processed_partitions"):
        for raw_path in manifest.get("files", {}).get(group, []):
            path = Path(raw_path)
            if not path.is_absolute():
                if not path.exists():
                    path = root.parent / path
            result.append(path)
    return result


def manifest_payload_digest(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key not in {"integrity", "dataset_fingerprint"}}
    return sha256_bytes(canonical_json_bytes(payload))


def audit_dataset(version: str, *, root: str | Path = "data") -> dict[str, Any]:
    lake = ParquetDataLake(root)
    manifest = lake.read_manifest(version)
    root_path = Path(root)
    file_rows = []
    for path in _resolved_files(manifest, root_path):
        exists = path.is_file()
        file_rows.append({
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": sha256_file(path) if exists else None,
        })
    file_rows.sort(key=lambda item: item["path"])
    file_index_digest = sha256_bytes(canonical_json_bytes(file_rows))
    manifest_digest = manifest_payload_digest(manifest)
    fingerprint = sha256_bytes(canonical_json_bytes({
        "dataset_version": version,
        "manifest_sha256": manifest_digest,
        "file_index_sha256": file_index_digest,
    }))

    requested = list(manifest.get("symbols_requested", []))
    succeeded = list(manifest.get("symbols_succeeded", []))
    quality = manifest.get("quality", {})
    raw_total = int(manifest.get("row_counts", {}).get("raw_total", 0))
    missing_total = sum(int(item.get("missing_minute_intervals", 0)) for item in quality.values())
    gap_rate = missing_total / (raw_total + missing_total) if raw_total + missing_total else 0.0
    firsts = [item.get("first_timestamp") for item in quality.values() if item.get("first_timestamp")]
    lasts = [item.get("last_timestamp") for item in quality.values() if item.get("last_timestamp")]

    span_days = 0.0
    if firsts and lasts:
        span_days = (datetime.fromisoformat(max(lasts)) - datetime.fromisoformat(min(firsts))).total_seconds() / 86400.0

    report = {
        "schema_version": 1,
        "dataset_version": version,
        "audited_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_fingerprint": fingerprint,
        "manifest_sha256": manifest_digest,
        "file_index_sha256": file_index_digest,
        "files_total": len(file_rows),
        "files_missing": [row["path"] for row in file_rows if not row["exists"]],
        "total_bytes": sum(int(row["size_bytes"] or 0) for row in file_rows),
        "symbols_requested": len(requested),
        "symbols_succeeded": len(succeeded),
        "symbol_success_rate": len(succeeded) / len(requested) if requested else 0.0,
        "failed_symbols": manifest.get("failed_symbols", {}),
        "raw_rows": raw_total,
        "processed_rows": int(manifest.get("row_counts", {}).get("processed_total", 0)),
        "missing_minute_intervals": missing_total,
        "missing_minute_rate": gap_rate,
        "earliest_timestamp": min(firsts) if firsts else None,
        "latest_timestamp": max(lasts) if lasts else None,
        "calendar_span_days": span_days,
        "context_symbols": manifest.get("context_symbols", []),
        "feature_count": len(manifest.get("features", [])),
        "quality": quality,
        "files": file_rows,
    }
    report["gates"] = dataset_gates(report)
    report["all_mandatory_gates_pass"] = all(gate["passed"] for gate in report["gates"] if gate["mandatory"])
    return report


def dataset_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    def gate(name: str, passed: bool, value: Any, threshold: Any, *, mandatory: bool = True) -> dict[str, Any]:
        return {"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "mandatory": mandatory}

    return [
        gate("all_manifest_files_exist", not report["files_missing"], len(report["files_missing"]), 0),
        gate("training_symbol_count", report["symbols_succeeded"] >= 50, report["symbols_succeeded"], ">=50"),
        gate("calendar_span_at_least_two_years", report["calendar_span_days"] >= 700, report["calendar_span_days"], ">=700 days"),
        gate("symbol_success_rate", report["symbol_success_rate"] >= 0.98, report["symbol_success_rate"], ">=0.98"),
        gate("processed_rows_nonzero", report["processed_rows"] > 0, report["processed_rows"], ">0"),
        gate("spy_context_present", "SPY" in report["context_symbols"], report["context_symbols"], "contains SPY"),
        gate("qqq_context_present", "QQQ" in report["context_symbols"], report["context_symbols"], "contains QQQ"),
        gate("feature_schema_nonempty", report["feature_count"] > 0, report["feature_count"], ">0"),
        gate("missing_minute_rate_below_2pct", report["missing_minute_rate"] < 0.02, report["missing_minute_rate"], "<0.02", mandatory=False),
    ]


def freeze_dataset(version: str, *, root: str | Path = "data", output: str | Path | None = None) -> dict[str, Any]:
    report = audit_dataset(version, root=root)
    if report["files_missing"]:
        raise FileNotFoundError(f"cannot freeze dataset with missing files: {report['files_missing'][:5]}")
    path = Path(output) if output else Path(root) / "manifests" / f"{version}.freeze.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("dataset_fingerprint") != report["dataset_fingerprint"]:
            raise RuntimeError("frozen dataset fingerprint changed; immutable dataset was modified")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/freeze an immutable Joe Bot research dataset")
    parser.add_argument("version")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = freeze_dataset(args.version, root=args.data_root, output=args.output) if args.freeze else audit_dataset(args.version, root=args.data_root)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
