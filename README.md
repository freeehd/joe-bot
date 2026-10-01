# Joe Bot — AI Quick Trader

Joe Bot is an experimental short-horizon US-equities quantitative research project. The repository is being rebuilt according to `AI_Quick_Trader_Master_Plan.md`.

## Safety state

- Research / paper analysis only.
- No automatic live trading is enabled.
- The Phase B/C/D research pipeline does not submit orders.
- Legacy broker scripts under `execution/` are not part of the offline unit-test suite.

## Current research architecture

```text
historical minute bars
        ↓
immutable raw Parquet snapshot
        ↓
data-quality validation
        ↓
Feature Engine V2 (58 causal features)
        ↓
SPY / QQQ / breadth market context
        ↓
triple-barrier WAIT / LONG / SHORT labels
        ↓
versioned processed dataset + manifest
        ↓
chronological purged train / calibration / test
        ↓
multiclass model-family benchmarks
        ↓
probability calibration + held-out metrics
```

The original v0.3 scanner remains available only for baseline compatibility. It should not be treated as evidence of profitability.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add **paper/data** Alpaca credentials. Never commit `.env`.

Optional model-family benchmarks:

```bash
pip install -r requirements-model-benchmarks.txt
```

## Build the Phase B/C historical lake

The research universe contains 50 liquid equities. SPY and QQQ are acquired separately as market-context symbols and are not silently mixed into a custom training universe.

Build two calendar years of regular-session minute history:

```bash
python -m research.build_dataset \
  --version v05-r50-2y-001 \
  --years 2 \
  --feed iex
```

Use `--feed sip` only when your market-data entitlement supports it and you explicitly want SIP history. The exact feed, adjustment mode, date range, code commit, feature schema, barrier parameters, class distribution, quality statistics, and chronological split periods are frozen in the dataset manifest.

Data layout:

```text
data/
  raw/
    datasets/<version>/bars/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/bars.parquet
  processed/
    datasets/<version>/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/training.parquet
  manifests/
    <version>.json
  experiments/
    ...
```

Dataset versions are immutable.

Inspect a dataset:

```bash
python -m research.report_dataset v05-r50-2y-001
```

Rebuild Feature Engine V2 + labels from the same immutable raw snapshot:

```bash
python -m research.rebuild_dataset \
  --source-version v05-r50-2y-001 \
  --new-version v05-r50-2y-labels-002
```

Add `--reuse-source-label-config` for an exact label-configuration rebuild.

## Feature Engine V2

The current V2 schema contains 58 causal features across:

- price / momentum,
- normalized trend and EMA structure,
- ATR / realized volatility / range expansion,
- volume and trade-count activity,
- historical exact-time-of-day relative volume,
- session/time features,
- SPY and QQQ context,
- relative strength,
- cross-sectional breadth.

Exact-time relative-volume baselines use only **earlier sessions** for that symbol. Returns, rolling calculations, and labels reset at New York session boundaries.

## Compare barrier regimes before training

Use the immutable raw snapshot to understand how label choices behave before spending compute on models:

```bash
python -m research.barrier_sweep \
  --source-version v05-r50-2y-001 \
  --targets 0.002,0.003,0.004 \
  --stops 0.001,0.0015,0.002 \
  --horizons 5,10,15,20 \
  --output data/experiments/barrier_sweep.csv
```

The sweep reports WAIT/LONG/SHORT rates, ambiguity, and no-resolution rates. It does **not** pretend that class balance alone determines profitability.

## Train the V0.5 XGBoost baseline

```bash
python -m models.train_v2 \
  --dataset-version v05-r50-2y-001
```

The V0.5 evaluation reports:

- LONG / SHORT precision and recall,
- macro F1,
- LONG / SHORT PR-AUC,
- multiclass ROC-AUC where defined,
- log loss,
- multiclass Brier score,
- expected calibration error,
- confidence buckets.

Model bundles record the source dataset version and SHA-256 digest of its manifest.

## Benchmark model families

Do not assume XGBoost wins:

```bash
python -m research.benchmark_models \
  --dataset-version v05-r50-2y-001 \
  --output data/experiments/model_benchmark.json
```

The harness always includes XGBoost and sklearn HistGradientBoosting. LightGBM and CatBoost are included automatically when installed. Every candidate receives the same frozen features and chronological train/calibration/test windows.

No model is promoted to production merely for having the best generic classification score. Later phases must compare **net expected value after realistic costs**.

## Data-quality policy

The builder is deliberately conservative:

- corporate-action adjustment mode is explicit and versioned,
- timestamps are normalized to UTC,
- research rows are filtered to 09:30–16:00 America/New_York,
- duplicate timestamps are resolved deterministically,
- malformed/non-positive price rows and negative-volume rows are dropped and counted,
- impossible OHLC rows are dropped and counted,
- missing minute intervals are measured but **never forward-filled**,
- raw snapshots are immutable per dataset version,
- benchmark/context requirements are explicit.

## Offline tests

```bash
python -m unittest discover -s tests -v
```

The tests are broker-safe and do not place orders.
