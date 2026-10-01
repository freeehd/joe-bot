"""Train and evaluate the Feature Engine V2 multiclass alpha model.

The command consumes an immutable dataset version; it never downloads market
data or submits orders. All preprocessing choices are read from the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from labels.triple_barrier import BarrierConfig, TradeLabel
from models.train_multiclass import chronological_purged_split
from research.storage import ParquetDataLake

MODEL_PATH = "data/models/xgboost_v2_calibrated.pkl"
METADATA_PATH = "data/models/xgboost_v2_calibrated.metadata.json"
CLASS_MAPPING = {0: "WAIT", 1: "LONG", 2: "SHORT"}


def build_xgboost_v2() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=800,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.80,
        colsample_bytree=0.75,
        min_child_weight=8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def confidence_bucket_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows: list[dict] = []
    edges = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        rows.append(
            {
                "low": low,
                "high": min(high, 1.0),
                "samples": int(mask.sum()),
                "accuracy": float((predicted[mask] == y_true[mask]).mean()),
                "avg_confidence": float(confidence[mask].mean()),
                "long_predictions": int((predicted[mask] == int(TradeLabel.LONG)).sum()),
                "short_predictions": int((predicted[mask] == int(TradeLabel.SHORT)).sum()),
            }
        )
    return rows


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    predictions = probabilities.argmax(axis=1)
    one_hot = label_binarize(y, classes=[0, 1, 2])

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "long_precision": float(precision_score(y, predictions, labels=[1], average="macro", zero_division=0)),
        "long_recall": float(recall_score(y, predictions, labels=[1], average="macro", zero_division=0)),
        "short_precision": float(precision_score(y, predictions, labels=[2], average="macro", zero_division=0)),
        "short_recall": float(recall_score(y, predictions, labels=[2], average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10": expected_calibration_error(y, probabilities, bins=10),
    }

    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    for class_id, name in ((1, "long"), (2, "short")):
        try:
            metrics[f"{name}_pr_auc"] = float(
                average_precision_score((y == class_id).astype(int), probabilities[:, class_id])
            )
        except ValueError:
            metrics[f"{name}_pr_auc"] = float("nan")

    metrics["classification_report"] = classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        target_names=["WAIT", "LONG", "SHORT"],
        zero_division=0,
        output_dict=True,
    )
    metrics["confidence_buckets"] = confidence_bucket_report(y, probabilities)
    return metrics


def load_dataset(version: str, *, root: str = "data") -> tuple[pd.DataFrame, dict, list[str]]:
    lake = ParquetDataLake(root)
    manifest = lake.read_manifest(version)
    dataset = lake.load_processed_dataset(version).sort_index()
    features = list(manifest["features"])
    missing = set(features + ["trade_label", "training_symbol"]).difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset {version!r} missing columns: {sorted(missing)}")
    if manifest.get("feature_engine") != "v2":
        raise ValueError(
            f"Dataset {version!r} uses feature engine {manifest.get('feature_engine')!r}, expected 'v2'"
        )
    return dataset, manifest, features


def train_dataset(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    calibration_method: str = "sigmoid",
):
    barrier_config = BarrierConfig(**manifest["label_parameters"])
    split = manifest.get("split_policy", {})
    train_fraction = float(split.get("train_fraction", 0.70))
    calibration_fraction = float(split.get("calibration_fraction", 0.15))

    train, calibration, test = chronological_purged_split(
        dataset,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        purge_bars=barrier_config.horizon_bars,
    )

    base = build_xgboost_v2()
    weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
    base.fit(
        train[feature_columns],
        train["trade_label"].astype(int),
        sample_weight=weights,
    )

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base),
        method=calibration_method,
    )
    calibrated.fit(
        calibration[feature_columns],
        calibration["trade_label"].astype(int),
    )

    probabilities = calibrated.predict_proba(test[feature_columns])
    metrics = evaluate_probabilities(test["trade_label"], probabilities)
    return calibrated, metrics, (train, calibration, test)


def manifest_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_bundle(
    model,
    *,
    dataset_version: str,
    manifest: dict,
    feature_columns: list[str],
    metrics: dict,
    output_path: str = MODEL_PATH,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": model,
        "version": "v0.5",
        "dataset_version": dataset_version,
        "dataset_manifest_sha256": manifest_digest(manifest),
        "feature_engine": "v2",
        "feature_columns": feature_columns,
        "class_mapping": CLASS_MAPPING,
        "label_parameters": manifest["label_parameters"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    joblib.dump(bundle, output_path)
    metadata_path = (
        METADATA_PATH if output_path == MODEL_PATH else f"{output_path}.metadata.json"
    )
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 76)
    print("V0.5 FEATURE ENGINE V2 — HELD-OUT TEST")
    print("=" * 76)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"LONG precision:  {metrics['long_precision']:.4f}")
    print(f"LONG recall:     {metrics['long_recall']:.4f}")
    print(f"SHORT precision: {metrics['short_precision']:.4f}")
    print(f"SHORT recall:    {metrics['short_recall']:.4f}")
    print(f"LONG PR-AUC:     {metrics['long_pr_auc']:.4f}")
    print(f"SHORT PR-AUC:    {metrics['short_pr_auc']:.4f}")
    print(f"Log loss:        {metrics['log_loss']:.4f}")
    print(f"Brier:           {metrics['multiclass_brier']:.4f}")
    print(f"ECE(10):         {metrics['ece_10']:.4f}")
    print("\nConfidence buckets:")
    for row in metrics["confidence_buckets"]:
        print(
            f"  {row['low']:.0%}-{row['high']:.0%}: "
            f"n={row['samples']:<7} acc={row['accuracy']:.3f} "
            f"conf={row['avg_confidence']:.3f} "
            f"L={row['long_predictions']} S={row['short_predictions']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Feature Engine V2 XGBoost")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--output", default=MODEL_PATH)
    args = parser.parse_args()

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    print(f"Loaded {len(dataset):,} rows, {len(features)} features from {args.dataset_version}")
    model, metrics, _ = train_dataset(
        dataset,
        manifest,
        features,
        calibration_method=args.calibration,
    )
    print_metrics(metrics)
    save_bundle(
        model,
        dataset_version=args.dataset_version,
        manifest=manifest,
        feature_columns=features,
        metrics=metrics,
        output_path=args.output,
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
