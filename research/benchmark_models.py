"""Benchmark multiple multiclass model families on one frozen V2 dataset.

Every candidate receives the same chronological train/calibration/test split and
the same feature matrix. This prevents model comparisons from being confounded
by different preprocessing or holdout windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from labels.triple_barrier import BarrierConfig
from models.train_multiclass import chronological_purged_split
from models.train_v2 import evaluate_probabilities, load_dataset


def available_candidates() -> dict[str, callable]:
    candidates: dict[str, callable] = {
        "xgboost": lambda: XGBClassifier(
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
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        candidates["catboost"] = lambda: CatBoostClassifier(
            loss_function="MultiClass",
            iterations=700,
            depth=7,
            learning_rate=0.03,
            l2_leaf_reg=4.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:
        pass

    return candidates


def benchmark_dataset(
    dataset,
    manifest: dict,
    feature_columns: list[str],
    *,
    candidate_names: list[str] | None = None,
    calibration_method: str = "sigmoid",
) -> list[dict]:
    barrier_config = BarrierConfig(**manifest["label_parameters"])
    split = manifest.get("split_policy", {})
    train, calibration, test = chronological_purged_split(
        dataset,
        train_fraction=float(split.get("train_fraction", 0.70)),
        calibration_fraction=float(split.get("calibration_fraction", 0.15)),
        purge_bars=barrier_config.horizon_bars,
    )

    factories = available_candidates()
    requested = candidate_names or list(factories)
    unknown = [name for name in requested if name not in factories]
    if unknown:
        raise ValueError(
            f"Unavailable candidate(s): {unknown}. Available: {sorted(factories)}"
        )

    weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
    results: list[dict] = []

    for name in requested:
        print(f"\nTraining {name}...")
        model = factories[name]()
        started = perf_counter()
        model.fit(
            train[feature_columns],
            train["trade_label"].astype(int),
            sample_weight=weights,
        )
        fit_seconds = perf_counter() - started

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model),
            method=calibration_method,
        )
        calibration_started = perf_counter()
        calibrated.fit(
            calibration[feature_columns],
            calibration["trade_label"].astype(int),
        )
        calibration_seconds = perf_counter() - calibration_started

        probabilities = calibrated.predict_proba(test[feature_columns])
        metrics = evaluate_probabilities(test["trade_label"], probabilities)
        result = {
            "model": name,
            "fit_seconds": float(fit_seconds),
            "calibration_seconds": float(calibration_seconds),
            "test_rows": int(len(test)),
            "feature_count": int(len(feature_columns)),
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"  macro_f1={metrics['macro_f1']:.4f} "
            f"LONG_PR={metrics['long_pr_auc']:.4f} "
            f"SHORT_PR={metrics['short_pr_auc']:.4f} "
            f"logloss={metrics['log_loss']:.4f} ECE={metrics['ece_10']:.4f}"
        )

    return results


def write_benchmark(results: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark V2 multiclass model families")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--output", default="data/experiments/model_benchmark.json")
    args = parser.parse_args()

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    print(
        f"Dataset {args.dataset_version}: {len(dataset):,} rows, "
        f"{len(features)} features"
    )
    print("Available candidates:", ", ".join(sorted(available_candidates())))
    results = benchmark_dataset(
        dataset,
        manifest,
        features,
        candidate_names=args.models or None,
        calibration_method=args.calibration,
    )
    write_benchmark(results, args.output)
    print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
