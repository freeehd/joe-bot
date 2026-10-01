"""Fit held-out post-hoc temperatures for a fine-tuned V0.8 Laya checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.laya_calibration import LayaCalibration, expected_calibration_error, fit_temperature, temperature_scale
from research.laya_infer import read_jsonl

ACTION_LABELS = ["LONG", "WAIT", "SHORT"]
QUALITY_LABELS = ["POOR", "FAIR", "GOOD", "EXCELLENT"]


def _matrix(records: list[dict], field: str, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    gold = []
    for record in records:
        prediction = record["prediction"]
        probabilities = prediction[field]
        answer_key = "action" if field == "action_probabilities" else "quality"
        label = str(record["answers"][answer_key]).upper()
        if label not in labels:
            continue
        probs.append([float(probabilities.get(name, 0.0)) for name in labels])
        gold.append(labels.index(label))
    if not probs:
        raise ValueError(f"No usable records for {field}")
    return np.asarray(probs, dtype=float), np.asarray(gold, dtype=int)


def fit_laya_calibration(records: list[dict], *, fitted_on: str | None = None) -> tuple[LayaCalibration, dict]:
    action_probs, action_gold = _matrix(records, "action_probabilities", ACTION_LABELS)
    quality_probs, quality_gold = _matrix(records, "quality_probabilities", QUALITY_LABELS)

    risk_probs = []
    risk_gold = []
    for record in records:
        p = float(record["prediction"]["risk_concern_probability"])
        risk_probs.append([1.0 - p, p])
        risk_gold.append(int(bool(record["answers"]["risk_concern"])))
    risk_probs = np.asarray(risk_probs, dtype=float)
    risk_gold = np.asarray(risk_gold, dtype=int)

    action_t = fit_temperature(action_probs, action_gold)
    quality_t = fit_temperature(quality_probs, quality_gold)
    risk_t = fit_temperature(risk_probs, risk_gold)
    calibration = LayaCalibration(
        action_temperature=action_t,
        quality_temperature=quality_t,
        risk_temperature=risk_t,
        fitted_on=fitted_on,
    )

    report = {
        "records": len(records),
        "action_temperature": action_t,
        "quality_temperature": quality_t,
        "risk_temperature": risk_t,
        "action_ece_before": expected_calibration_error(action_probs, action_gold),
        "action_ece_after": expected_calibration_error(temperature_scale(action_probs, action_t), action_gold),
        "quality_ece_before": expected_calibration_error(quality_probs, quality_gold),
        "quality_ece_after": expected_calibration_error(temperature_scale(quality_probs, quality_t), quality_gold),
        "risk_ece_before": expected_calibration_error(risk_probs, risk_gold),
        "risk_ece_after": expected_calibration_error(temperature_scale(risk_probs, risk_t), risk_gold),
    }
    return calibration, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit held-out V0.8 Laya calibration")
    parser.add_argument("--predictions", required=True, help="Validation predictions JSONL from research.laya_infer")
    parser.add_argument("--output", default="data/laya/v08/calibration.json")
    parser.add_argument("--report", default="data/laya/v08/calibration_report.json")
    args = parser.parse_args()

    records = read_jsonl(args.predictions)
    calibration, report = fit_laya_calibration(records, fitted_on=args.predictions)
    calibration.save(args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved calibration: {args.output}")


if __name__ == "__main__":
    main()
