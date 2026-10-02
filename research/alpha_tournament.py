"""Leakage-safe Victory Sprint 2 alpha research utilities.

Selection never uses the final outer test partition.  The chronology is:

    outer train -> inner fit/calibration -> outer calibration (selection)
                                      ↓
                         pre-final-test walk-forward
                                      ↓
                   freeze multiple challenger finalists
                                      ↓
                    one-time outer test diagnostics

Outer-test metrics are reported for audit only and are never used to choose the
finalist set written by this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

from features.v2 import (
    FEATURE_COLUMNS_V2,
    MARKET_CONTEXT_FEATURES,
    TIME_FEATURES,
    TREND_FEATURES,
    VOLATILITY_FEATURES,
)
from labels.triple_barrier import BarrierConfig
from models.train_multiclass import chronological_purged_split
from models.train_v2 import CLASS_MAPPING, evaluate_probabilities, load_dataset, manifest_digest
from research.benchmark_models import available_candidates
from research.walk_forward import WalkForwardConfig, generate_windows, slice_window


BENCHMARK_RETURN_FEATURES = [
    "spy_return_1m", "spy_return_5m", "spy_return_15m",
    "qqq_return_1m", "qqq_return_5m", "qqq_return_15m",
]
BREADTH_FEATURES = ["breadth_up_1m"]
RELATIVE_STRENGTH_FEATURES = ["relative_strength_spy_5m", "relative_strength_qqq_5m"]
TIME_NORMALIZED_ACTIVITY_FEATURES = ["tod_relative_volume", "tod_relative_trade_count"]


ABLATION_GROUPS: dict[str, list[str]] = {
    "market_benchmarks": BENCHMARK_RETURN_FEATURES,
    "breadth": BREADTH_FEATURES,
    "relative_strength": RELATIVE_STRENGTH_FEATURES,
    "time_normalized_activity": TIME_NORMALIZED_ACTIVITY_FEATURES,
    "trend": list(TREND_FEATURES),
    "volatility": list(VOLATILITY_FEATURES),
    "session_time": list(TIME_FEATURES),
}


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    calibration_method: str
    feature_columns: tuple[str, ...]
    selection_score: float

    @property
    def candidate_id(self) -> str:
        return f"{self.model_name}-{self.calibration_method}"


def _require_three_classes(frame: pd.DataFrame, name: str) -> None:
    present = set(int(value) for value in frame["trade_label"].unique())
    missing = {0, 1, 2}.difference(present)
    if missing:
        raise ValueError(f"{name} partition missing classes {sorted(missing)}")


def outer_split(dataset: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    barrier = BarrierConfig(**manifest["label_parameters"])
    split = manifest.get("split_policy", {})
    train, calibration, final_test = chronological_purged_split(
        dataset,
        train_fraction=float(split.get("train_fraction", 0.70)),
        calibration_fraction=float(split.get("calibration_fraction", 0.15)),
        purge_bars=barrier.horizon_bars,
    )
    for name, frame in (("outer_train", train), ("outer_calibration", calibration), ("outer_final_test", final_test)):
        _require_three_classes(frame, name)
    return train, calibration, final_test


def inner_fit_calibration(outer_train: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split outer train again, reserving its tail from fitting.

    The third return of chronological_purged_split is intentionally discarded;
    outer calibration remains the actual model/feature selection holdout.
    """
    barrier = BarrierConfig(**manifest["label_parameters"])
    fit, inner_calibration, _ = chronological_purged_split(
        outer_train,
        train_fraction=0.72,
        calibration_fraction=0.14,
        purge_bars=barrier.horizon_bars,
    )
    _require_three_classes(fit, "inner_fit")
    _require_three_classes(inner_calibration, "inner_calibration")
    return fit, inner_calibration


def selection_score(metrics: dict) -> float:
    """Classification/calibration score used only before the final test boundary."""
    directional_pr = 0.5 * (float(metrics["long_pr_auc"]) + float(metrics["short_pr_auc"]))
    return float(
        0.45 * float(metrics["macro_f1"])
        + 0.35 * directional_pr
        - 0.10 * float(metrics["log_loss"])
        - 0.10 * float(metrics["ece_10"])
    )


def fit_calibrated(
    factory: Callable[[], object],
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    *,
    calibration_method: str,
):
    model = factory()
    weights = compute_sample_weight("balanced", fit["trade_label"].astype(int))
    model.fit(fit[feature_columns], fit["trade_label"].astype(int), sample_weight=weights)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=calibration_method)
    calibrated.fit(calibration[feature_columns], calibration["trade_label"].astype(int))
    return calibrated


def evaluate_candidate(
    factory: Callable[[], object],
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    feature_columns: list[str],
    *,
    calibration_method: str = "sigmoid",
) -> dict:
    started = perf_counter()
    model = fit_calibrated(
        factory, fit, calibration, feature_columns, calibration_method=calibration_method
    )
    fit_seconds = perf_counter() - started
    probabilities = model.predict_proba(selection[feature_columns])
    metrics = evaluate_probabilities(selection["trade_label"], probabilities)
    return {
        "feature_count": len(feature_columns),
        "fit_and_calibration_seconds": float(fit_seconds),
        "selection_rows": int(len(selection)),
        "selection_score": selection_score(metrics),
        "metrics": metrics,
    }


def run_feature_ablation(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    model_name: str = "xgboost",
    calibration_method: str = "sigmoid",
) -> list[dict]:
    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"model {model_name!r} unavailable; available={sorted(factories)}")
    outer_train, selection, _ = outer_split(dataset, manifest)
    fit, inner_calibration = inner_fit_calibration(outer_train, manifest)

    experiments: list[tuple[str, list[str]]] = [("full", list(feature_columns))]
    for name, group in ABLATION_GROUPS.items():
        group_set = set(group)
        kept = [column for column in feature_columns if column not in group_set]
        experiments.append((f"without_{name}", kept))

    results: list[dict] = []
    baseline_score: float | None = None
    for name, columns in experiments:
        result = evaluate_candidate(
            factories[model_name], fit, inner_calibration, selection, columns,
            calibration_method=calibration_method,
        )
        if name == "full":
            baseline_score = result["selection_score"]
        result.update({
            "experiment": name,
            "model": model_name,
            "calibration": calibration_method,
            "removed_features": [] if name == "full" else sorted(set(feature_columns).difference(columns)),
            "final_test_used": False,
        })
        results.append(result)
    assert baseline_score is not None
    for row in results:
        row["score_delta_vs_full"] = float(row["selection_score"] - baseline_score)
    return sorted(results, key=lambda row: row["selection_score"], reverse=True)


def run_model_tournament(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    model_names: Iterable[str] | None = None,
    calibration_methods: Iterable[str] = ("sigmoid", "isotonic"),
) -> dict:
    factories = available_candidates()
    requested = list(model_names) if model_names is not None else [
        "xgboost", "hist_gradient_boosting", "lightgbm", "catboost"
    ]
    available = [name for name in requested if name in factories]
    unavailable = [name for name in requested if name not in factories]
    if not available:
        raise RuntimeError("none of the requested model families are available")

    outer_train, selection, final_test = outer_split(dataset, manifest)
    fit, inner_calibration = inner_fit_calibration(outer_train, manifest)
    rows: list[dict] = []
    for model_name in available:
        for calibration in calibration_methods:
            result = evaluate_candidate(
                factories[model_name], fit, inner_calibration, selection, feature_columns,
                calibration_method=calibration,
            )
            result.update({
                "candidate_id": f"{model_name}-{calibration}",
                "model": model_name,
                "calibration": calibration,
                "final_test_used": False,
            })
            rows.append(result)
    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    return {
        "requested_models": requested,
        "available_models": available,
        "unavailable_models": unavailable,
        "selection_partition": {
            "rows": int(len(selection)),
            "start": pd.Timestamp(selection.index.min()).isoformat(),
            "end": pd.Timestamp(selection.index.max()).isoformat(),
        },
        "final_test_partition": {
            "rows": int(len(final_test)),
            "start": pd.Timestamp(final_test.index.min()).isoformat(),
            "end": pd.Timestamp(final_test.index.max()).isoformat(),
            "used_for_ranking": False,
        },
        "results": rows,
    }


def run_classification_walk_forward(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    model_name: str,
    calibration_method: str,
    config: WalkForwardConfig | None = None,
) -> dict:
    """Classification-only walk-forward that stops before outer final test."""
    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"model {model_name!r} unavailable")
    config = config or WalkForwardConfig()
    barrier = BarrierConfig(**manifest["label_parameters"])
    outer_train, outer_calibration, final_test = outer_split(dataset, manifest)
    pretest = pd.concat([outer_train, outer_calibration]).sort_index()
    final_start = pd.Timestamp(final_test.index.min())
    windows = [
        window for window in generate_windows(pd.DatetimeIndex(pretest.index.unique()), config)
        if window.test_end <= final_start
    ]
    results: list[dict] = []
    for window in windows:
        train, calibration, test = slice_window(pretest, window, purge_bars=barrier.horizon_bars)
        if train.empty or calibration.empty or test.empty:
            continue
        try:
            _require_three_classes(train, "wf_train")
            _require_three_classes(calibration, "wf_calibration")
            _require_three_classes(test, "wf_test")
        except ValueError:
            continue
        model = fit_calibrated(
            factories[model_name], train, calibration, feature_columns,
            calibration_method=calibration_method,
        )
        metrics = evaluate_probabilities(test["trade_label"], model.predict_proba(test[feature_columns]))
        results.append({"window": window.to_dict(), "rows": int(len(test)), "metrics": metrics})
    if not results:
        raise RuntimeError("no complete pre-final-test walk-forward windows")

    keys = ["macro_f1", "long_pr_auc", "short_pr_auc", "log_loss", "multiclass_brier", "ece_10"]
    averages = {key: float(np.mean([row["metrics"][key] for row in results])) for key in keys}
    return {
        "model": model_name,
        "calibration": calibration_method,
        "feature_count": len(feature_columns),
        "final_test_used": False,
        "final_test_start": final_start.isoformat(),
        "windows_completed": len(results),
        "average_metrics": averages,
        "windows": results,
    }


def freeze_challenger(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    model_name: str,
    calibration_method: str,
    selection_rank: int,
    selection_metrics: dict,
    output_path: str | Path,
) -> dict:
    """Refit/calibrate a selected challenger and report one-time final-test diagnostics.

    Final-test metrics are not returned to the selection/ranking process.  Multiple
    pre-selected finalists should be carried to Sprint 3 rather than re-ranked here.
    """
    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"model {model_name!r} unavailable")
    outer_train, outer_calibration, final_test = outer_split(dataset, manifest)
    model = fit_calibrated(
        factories[model_name], outer_train, outer_calibration, feature_columns,
        calibration_method=calibration_method,
    )
    probabilities = model.predict_proba(final_test[feature_columns])
    final_metrics = evaluate_probabilities(final_test["trade_label"], probabilities)
    bundle = {
        "model": model,
        "version": "victory-sprint2",
        "model_family": model_name,
        "calibration_method": calibration_method,
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_sha256": manifest_digest(manifest),
        "feature_engine": "v2",
        "feature_columns": feature_columns,
        "class_mapping": CLASS_MAPPING,
        "label_parameters": manifest["label_parameters"],
        "selection_rank": int(selection_rank),
        "selection_metrics": selection_metrics,
        "final_test_metrics_audit_only": final_metrics,
        "final_test_used_for_selection": False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return {"bundle": str(output), "metadata": str(metadata_path), "final_test_metrics": final_metrics}
