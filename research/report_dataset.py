"""Print a compact report from a versioned dataset manifest."""

from __future__ import annotations

import argparse

from research.storage import ParquetDataLake


def render_manifest(manifest: dict) -> str:
    lines = [
        f"Dataset: {manifest['dataset_version']}",
        f"Provider: {manifest['source']['provider']} / {manifest['source'].get('feed')}",
        f"Range: {manifest['date_range']['start']} -> {manifest['date_range']['end']}",
        f"Symbols: {len(manifest['symbols_succeeded'])}/{len(manifest['symbols_requested'])}",
        f"Raw rows: {manifest['row_counts']['raw_total']:,}",
        f"Processed rows: {manifest['row_counts']['processed_total']:,}",
        "Classes:",
    ]
    for label, stats in manifest.get("class_distribution", {}).items():
        lines.append(f"  {label:<5} {stats['count']:>10,} {stats['percent']:>8.3f}%")

    failed = manifest.get("failed_symbols", {})
    if failed:
        lines.append("Failed symbols:")
        for symbol, reason in sorted(failed.items()):
            lines.append(f"  {symbol}: {reason}")

    quality = manifest.get("quality", {})
    if quality:
        gaps = sum(item.get("missing_minute_intervals", 0) for item in quality.values())
        duplicates = sum(item.get("duplicate_rows", 0) for item in quality.values())
        invalid = sum(
            item.get("invalid_numeric_rows", 0) + item.get("invalid_ohlc_rows", 0)
            for item in quality.values()
        )
        lines.extend(
            [
                "Quality totals:",
                f"  missing minute intervals: {gaps:,}",
                f"  duplicate rows:           {duplicates:,}",
                f"  invalid rows dropped:     {invalid:,}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a Phase B dataset manifest")
    parser.add_argument("version")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    manifest = ParquetDataLake(args.data_root).read_manifest(args.version)
    print(render_manifest(manifest))


if __name__ == "__main__":
    main()
