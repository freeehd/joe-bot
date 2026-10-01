"""V0.4 multiclass alpha training pipeline.

This module implements the Master Plan's first intelligence milestone:
triple-barrier LONG/WAIT/SHORT labels, chronological purged validation, and
probability calibration. It intentionally does not submit trades.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config.settings import (
    LABEL_ATR_PERIOD,
    LABEL_ATR_STOP_MULTIPLIER,
    LABEL_ATR_TARGET_MULTIPLIER,
    LABEL_HORIZON_BARS,
    LABEL_STOP_PCT,
    LABEL_TARGET_PCT,
    LABEL_USE_ATR,
    V04_CALIBRATION_FRACTION,
    V04_TRAIN_FRACTION,
    V04_TRAINING_DAYS,
)
from features.feature_engine import create_features
from features.schema import FEATURE_COLUMNS
from labels.triple_barrier import BarrierConfig, TradeLabel, create_multiclass_labels
from market.universe import SYMBOLS


MODEL_PATH = "data/models/xgboost_multiclass_calibrated.pkl"
METADATA_PATH = "data/models/xgboost_multiclass_calibrated.metadata.json"


def barrier_config_from_settings() -> BarrierConfig:
    return BarrierConfig(
        horizon_bars=LABEL_HORIZON_BARS,
        target_pct=LABEL_TARGET_PCT,
        stop_pct=LABEL_STOP_PCT,
        use_atr=LABEL_USE_ATR,
        atr_period=LABEL_ATR_PERIOD,
        atr_target_multiplier=LABEL_ATR_TARGET_MULTIPLIER,
        atr_stop_multiplier=LABEL_ATR_STOP_MULTIPLIER,
    )


def build_training_frame(
    raw_df: pd.DataFrame,
    *,
    symbol: str,
    barrier_config: BarrierConfig,
) -> pd.DataFrame:
    """Turn one symbol's raw minute bars into leakage-safe training rows."""

    df = raw_df.copy()
    if "symbol" in df.index.names:
        df = df.reset_index(level="symbol", drop=True)

    df = df.sort_index()
    df = create_features(df, session_aware=True)
    df = create_multiclass_labels(df, barrier_config, respect_sessions=True)
    df["training_symbol"] = symbol

    required = FEATURE_COLUMNS + ["trade_label"]
    df = df[df["label_valid"]].dropna(subset=required)
    df["trade_label"] = df["trade_label"].astype(int)
    return df


def download_training_data(
    *,
    symbols: list[str] | None = None,
    days: int = V04_TRAINING_DAYS,
    barrier_config: BarrierConfig | None = None,
) -> pd.DataFrame:
    """Download and combine a multi-stock V0.4 dataset from Alpaca."""

    # Deferred import keeps offline unit tests independent of broker packages.
    from market.alpaca_client import get_bars

    symbols = symbols or SYMBOLS
    barrier_config = barrier_config or barrier_config_from_settings()
    frames: list[pd.DataFrame] = []

    print("\n" + "=" * 72)
    print("BUILDING V0.4 MULTICLASS TRAINING DATASET")
    print("=" * 72)

    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...", end=" ")
            raw = get_bars(symbol, days=days)
            if raw is None or len(raw) == 0:
                print("NO DATA")
                continue

            prepared = build_training_frame(
                raw,
                symbol=symbol,
                barrier_config=barrier_config,
            )
            if prepared.empty:
                print("NO USABLE ROWS")
                continue

            frames.append(prepared)
            print(f"{len(prepared):,} rows")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not frames:
        raise RuntimeError("No V0.4 training data was produced.")

    return pd.concat(frames, axis=0).sort_index()


def chronological_purged_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = V04_TRAIN_FRACTION,
    calibration_fraction: float = V04_CALIBRATION_FRACTION,
    purge_bars: int = LABEL_HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split while purging label horizon before boundaries."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Training frame must use a DatetimeIndex")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train + calibration fractions must leave a test set")
    if purge_bars < 0:
        raise ValueError("purge_bars must be >= 0")

    timestamps = pd.Index(df.index.unique()).sort_values()
    n_times = len(timestamps)
    train_boundary = int(n_times * train_fraction)
    calibration_boundary = int(n_times * (train_fraction + calibration_fraction))

    train_end = train_boundary - purge_bars
    calibration_end = calibration_boundary - purge_bars
    if train_end <= 0 or calibration_end <= train_boundary:
        raise ValueError("Dataset is too small for requested split and purge")

    train_times = timestamps[:train_end]
    calibration_times = timestamps[train_boundary:calibration_end]
    test_times = timestamps[calibration_boundary:]

    train = df.loc[df.index.isin(train_times)].copy()
    calibration = df.loc[df.index.isin(calibration_times)].copy()
    test = df.loc[df.index.isin(test_times)].copy()

    if train.empty or calibration.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")

    return train, calibration, test


def build_base_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def _probability_matrix(model, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])
    if probabilities.shape[1] != 3:
        raise RuntimeError("Expected three calibrated class probabilities")
    return probabilities


def print_distribution(name: str, frame: pd.DataFrame) -> None:
    counts = frame["trade_label"].value_counts().sort_index()
    percentages = frame["trade_label"].value_counts(normalize=True).sort_index() * 100
    print(f"\n{name} LABEL DISTRIBUTION")
    print(pd.DataFrame({"count": counts, "percent": percentages.round(2)}))


def print_confidence_buckets(y_true: pd.Series, probabilities: np.ndarray) -> None:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    report = pd.DataFrame(
        {
            "actual": y_true.to_numpy(),
            "predicted": predicted,
            "confidence": confidence,
        }
    )
    report["correct"] = report["actual"] == report["predicted"]
    report["bucket"] = pd.cut(
        report["confidence"],
        bins=[0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
        include_lowest=True,
    )
    summary = report.groupby("bucket", observed=True).agg(
        samples=("correct", "size"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean"),
    )
    print("\nOUT-OF-SAMPLE CONFIDENCE BUCKETS")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))


def evaluate_model(model, test: pd.DataFrame) -> dict[str, float]:
    X_test = test[FEATURE_COLUMNS]
    y_test = test["trade_label"].astype(int)
    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)

    print("\n" + "=" * 72)
    print("V0.4 OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 72)
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["WAIT", "LONG", "SHORT"],
            zero_division=0,
        )
    )

    metrics: dict[str, float] = {
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "accuracy": float((predictions == y_test.to_numpy()).mean()),
    }

    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    print(f"Log loss:          {metrics['log_loss']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"ROC-AUC OVR macro: {metrics['roc_auc_ovr_macro']:.4f}")
    print_confidence_buckets(y_test, probabilities)
    return metrics


def train_v04(dataset: pd.DataFrame, barrier_config: BarrierConfig):
    train, calibration, test = chronological_purged_split(
        dataset,
        purge_bars=barrier_config.horizon_bars,
    )

    print_distribution("TRAIN", train)
    print_distribution("CALIBRATION", calibration)
    print_distribution("TEST", test)

    base_model = build_base_model()
    train_weights = compute_sample_weight(
        class_weight="balanced",
        y=train["trade_label"].astype(int),
    )

    print("\nTraining multiclass XGBoost...")
    base_model.fit(
        train[FEATURE_COLUMNS],
        train["trade_label"].astype(int),
        sample_weight=train_weights,
    )

    print("Calibrating probabilities on a later chronological holdout...")
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model),
        method="sigmoid",
    )
    calibrated_model.fit(
        calibration[FEATURE_COLUMNS],
        calibration["trade_label"].astype(int),
    )

    metrics = evaluate_model(calibrated_model, test)
    return calibrated_model, metrics, (train, calibration, test)


def save_model_bundle(
    model,
    *,
    barrier_config: BarrierConfig,
    metrics: dict[str, float],
    dataset: pd.DataFrame,
) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_mapping": {
            int(TradeLabel.WAIT): "WAIT",
            int(TradeLabel.LONG): "LONG",
            int(TradeLabel.SHORT): "SHORT",
        },
        "barrier_config": asdict(barrier_config),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(dataset)),
        "metrics": metrics,
        "version": "v0.4",
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {key: value for key, value in bundle.items() if key != "model"}
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)

    print(f"\nSaved calibrated V0.4 model: {MODEL_PATH}")
    print(f"Saved metadata:              {METADATA_PATH}")


def main() -> None:
    barrier_config = barrier_config_from_settings()
    dataset = download_training_data(barrier_config=barrier_config)
    print(f"\nTOTAL V0.4 ROWS: {len(dataset):,}")

    model, metrics, _ = train_v04(dataset, barrier_config)
    save_model_bundle(
        model,
        barrier_config=barrier_config,
        metrics=metrics,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
