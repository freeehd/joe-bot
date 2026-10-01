"""Run a Laya checkpoint over generated V0.8 JSONL examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.laya_engine import LayaEngine


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def infer_records(
    records: list[dict],
    *,
    engine: LayaEngine,
    batch_size: int = 64,
) -> list[dict]:
    output: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        states = [record["state"] for record in batch]
        decisions = engine.evaluate_states(states, batch_size=batch_size)
        for source, decision in zip(batch, decisions):
            output.append(
                {
                    "prediction": decision.to_dict(),
                    "answers": source.get("answers", {}),
                    "metadata": source.get("metadata", {}),
                }
            )
        print(f"Laya inference: {min(start + len(batch), len(records)):,}/{len(records):,}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Laya inference on V0.8 JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Fine-tuned Laya checkpoint/repository id")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--head-max-len", type=int, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    engine = LayaEngine(
        args.model,
        device=args.device,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
    )
    predictions = infer_records(records, engine=engine, batch_size=args.batch_size)
    write_jsonl(predictions, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
