"""Victory Sprint 4 walk-forward comparison: core alpha vs specialists vs meta-model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

from backtest.engine import ExecutionConfig, TradeConfig, run_signal_backtest
from backtest.metrics import summarize_trades, trades_to_frame
from backtest.signals import probability_frame
from labels.triple_barrier import BarrierConfig
from models.train_v2 import evaluate_probabilities
from research.alpha_tournament import outer_split
from research.benchmark_models import available_candidates
from research.regime_analysis import trade_performance_by_regime
from research.storage import ParquetDataLake
from research.walk_forward import WalkForwardConfig, generate_windows, slice_window
from strategy.meta_model import SpecialistMetaModel
from strategy.regime import RegimeEngine
from strategy.specialists import default_specialists


@dataclass(frozen=True)
class IntelligenceExpansionThresholds:
    min_expectancy_improvement_bps: float = 0.0
    min_positive_window_rate_delta: float = 0.0
    min_regimes_with_nonnegative_ev: int = 2


def _trade_config(manifest: dict) -> TradeConfig:
    barrier = BarrierConfig(**manifest["label_parameters"])
    if barrier.use_atr:
        raise NotImplementedError("Sprint 4 execution comparison currently requires fixed-percent barriers")
    return TradeConfig(barrier.target_pct, barrier.stop_pct, barrier.horizon_bars)


def _classes(frame: pd.DataFrame) -> bool:
    return {0, 1, 2}.issubset(set(int(v) for v in frame["trade_label"].unique()))


def _attach_regime_to_trades(trades: pd.DataFrame, signals: pd.DataFrame, engine: RegimeEngine) -> pd.DataFrame:
    if trades.empty:
        return trades
    lookup: dict[tuple[str, pd.Timestamp], str] = {}
    for ts, row in signals.iterrows():
        lookup[(str(row["training_symbol"]), pd.Timestamp(ts))] = engine.infer_row(row).label
    result = trades.copy()
    result["regime"] = [lookup.get((str(row.symbol), pd.Timestamp(row.signal_time))) for row in result.itertuples()]
    return result


def _signals_from_specialist(test: pd.DataFrame, specialist, regime_engine: RegimeEngine) -> pd.DataFrame:
    probs = []
    for _, row in test.iterrows():
        regime = regime_engine.infer_row(row)
        signal = specialist.evaluate(row, regime)
        probs.append([signal.p_wait, signal.p_long, signal.p_short])
    return probability_frame(test, np.asarray(probs, dtype=float), classes=(0, 1, 2))


def _fit_core(factory, train, calibration, features, method):
    base = factory()
    weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
    base.fit(train[features], train["trade_label"].astype(int), sample_weight=weights)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    calibrated.fit(calibration[features], calibration["trade_label"].astype(int))
    return calibrated


def _summary(trades: pd.DataFrame) -> dict:
    return summarize_trades(trades if not trades.empty else pd.DataFrame())


def run_intelligence_expansion_walk_forward(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    lake: ParquetDataLake,
    model_name: str,
    calibration_method: str,
    walk_forward_config: WalkForwardConfig | None = None,
    execution_config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    wf = walk_forward_config or WalkForwardConfig()
    execution = execution_config or ExecutionConfig()
    trade = _trade_config(manifest)
    barrier = BarrierConfig(**manifest["label_parameters"])
    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"model {model_name!r} unavailable")

    outer_train, outer_calibration, final_test = outer_split(dataset, manifest)
    pretest = pd.concat([outer_train, outer_calibration]).sort_index()
    final_start = pd.Timestamp(final_test.index.min())
    windows = [w for w in generate_windows(pd.DatetimeIndex(pretest.index.unique()), wf) if w.test_end <= final_start]
    raw_version = manifest.get("raw_snapshot_version", manifest["dataset_version"])
    regime_engine = RegimeEngine()
    specialists = default_specialists()

    def loader(symbol: str) -> pd.DataFrame:
        return lake.load_raw_bars(raw_version, symbol)

    all_trades: dict[str, list[pd.DataFrame]] = {"core": [], "meta": [], **{s.name: [] for s in specialists}}
    results = []
    for window in windows:
        train, calibration, test = slice_window(pretest, window, purge_bars=barrier.horizon_bars)
        if train.empty or calibration.empty or test.empty or not all(_classes(x) for x in (train, calibration, test)):
            continue
        core = _fit_core(factories[model_name], train, calibration, feature_columns, calibration_method)
        core_probs = core.predict_proba(test[feature_columns])
        core_signals = probability_frame(test, core_probs, classes=core.classes_)
        core_trades = trades_to_frame(run_signal_backtest(core_signals, raw_loader=loader, trade_config=trade, execution_config=execution, confidence_threshold=wf.confidence_threshold))
        core_trades = _attach_regime_to_trades(core_trades, core_signals, regime_engine)
        if not core_trades.empty:
            core_trades["walk_forward_window"] = window.number
            all_trades["core"].append(core_trades)

        meta = SpecialistMetaModel(regime_engine=regime_engine).fit(train, calibration, method=calibration_method)
        meta_probs = meta.predict_proba(test)
        meta_signals = probability_frame(test, meta_probs, classes=meta.classes_)
        meta_trades = trades_to_frame(run_signal_backtest(meta_signals, raw_loader=loader, trade_config=trade, execution_config=execution, confidence_threshold=wf.confidence_threshold))
        meta_trades = _attach_regime_to_trades(meta_trades, meta_signals, regime_engine)
        if not meta_trades.empty:
            meta_trades["walk_forward_window"] = window.number
            all_trades["meta"].append(meta_trades)

        specialist_metrics = {}
        for specialist in specialists:
            signals = _signals_from_specialist(test, specialist, regime_engine)
            trades = trades_to_frame(run_signal_backtest(signals, raw_loader=loader, trade_config=trade, execution_config=execution, confidence_threshold=wf.confidence_threshold))
            trades = _attach_regime_to_trades(trades, signals, regime_engine)
            if not trades.empty:
                trades["walk_forward_window"] = window.number
                all_trades[specialist.name].append(trades)
            specialist_metrics[specialist.name] = _summary(trades)

        results.append({
            "window": window.to_dict(),
            "core_classification": evaluate_probabilities(test["trade_label"], core_probs),
            "meta_classification": evaluate_probabilities(test["trade_label"], meta_probs),
            "core_trades": _summary(core_trades),
            "meta_trades": _summary(meta_trades),
            "specialists": specialist_metrics,
        })

    if not results:
        raise RuntimeError("no complete pre-final-test Sprint 4 walk-forward windows")

    combined = {}
    for name, frames in all_trades.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined[name] = {
            "trade_metrics": _summary(frame),
            "regimes": trade_performance_by_regime(frame),
            "trade_records": frame.to_dict(orient="records") if not frame.empty else [],
        }
    core_ev = float(combined["core"]["trade_metrics"].get("expectancy_bps", 0.0))
    meta_ev = float(combined["meta"]["trade_metrics"].get("expectancy_bps", 0.0))
    core_positive = np.mean([row["core_trades"].get("avg_net_return", 0.0) > 0 for row in results])
    meta_positive = np.mean([row["meta_trades"].get("avg_net_return", 0.0) > 0 for row in results])
    return {
        "dataset_version": manifest["dataset_version"],
        "model": model_name,
        "calibration": calibration_method,
        "walk_forward_config": asdict(wf),
        "execution_config": asdict(execution),
        "final_test_used": False,
        "final_test_start": final_start.isoformat(),
        "windows_completed": len(results),
        "core_positive_window_rate": float(core_positive),
        "meta_positive_window_rate": float(meta_positive),
        "expectancy_improvement_bps": meta_ev - core_ev,
        "combined": combined,
        "windows": results,
    }


def build_intelligence_expansion_gate(result: dict, thresholds: IntelligenceExpansionThresholds | None = None) -> dict:
    thresholds = thresholds or IntelligenceExpansionThresholds()
    core = result["combined"]["core"]["trade_metrics"]
    meta = result["combined"]["meta"]["trade_metrics"]
    delta = float(result["expectancy_improvement_bps"])
    window_delta = float(result["meta_positive_window_rate"] - result["core_positive_window_rate"])
    healthy_regimes = sum(1 for row in result["combined"]["meta"]["regimes"] if row["expectancy_bps"] >= 0)
    gates = [
        {"name": "meta_expectancy_beats_core", "passed": delta > thresholds.min_expectancy_improvement_bps, "value": delta, "threshold": f">{thresholds.min_expectancy_improvement_bps}"},
        {"name": "meta_window_stability_not_worse", "passed": window_delta >= thresholds.min_positive_window_rate_delta, "value": window_delta, "threshold": f">={thresholds.min_positive_window_rate_delta}"},
        {"name": "meta_net_expectancy_positive", "passed": float(meta.get("expectancy_bps", 0.0)) > 0, "value": meta.get("expectancy_bps", 0.0), "threshold": ">0"},
        {"name": "regime_breadth", "passed": healthy_regimes >= thresholds.min_regimes_with_nonnegative_ev, "value": healthy_regimes, "threshold": f">={thresholds.min_regimes_with_nonnegative_ev}"},
    ]
    passed = all(g["passed"] for g in gates)
    return {
        "gate": "INTELLIGENCE EXPANSION",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": asdict(thresholds),
        "gates": gates,
        "core_expectancy_bps": core.get("expectancy_bps", 0.0),
        "meta_expectancy_bps": meta.get("expectancy_bps", 0.0),
        "expectancy_improvement_bps": delta,
        "promotion_instruction": (
            "Freeze the regime/specialist/meta challenger for one-time final-test audit."
            if passed else
            "Do not promote Sprint 4 intelligence. Keep the validated core alpha and investigate specialist/regime failures."
        ),
    }
