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

## V0.6 realistic backtesting + walk-forward

V0.6 evaluates model probabilities as **executable trades**, not as candle-close labels.
By default a signal created at minute `t` enters no earlier than the next bar open,
pays half the configured spread plus adverse slippage on both entry and exit,
rounds fills adversely to the minimum tick, exits on target/stop/time, and never
turns a near-close signal into an accidental overnight entry. If target and stop
are both touched inside one minute bar, the conservative default assumes the stop
was hit first.

Run rolling 6-month train / 1-month calibration / 1-month test windows:

```bash
python -m research.walk_forward \
  --dataset-version v05-r50-2y-001 \
  --model xgboost \
  --train-months 6 \
  --calibration-months 1 \
  --test-months 1 \
  --confidence 0.60 \
  --spread-bps 4 \
  --slippage-bps 2 \
  --entry-delay-bars 1 \
  --stress-execution
```

Each window independently fits and calibrates the model, then sends only the
untouched test-period signals to the event-driven simulator. Reports include
classification quality and actual simulated trade quality side by side:

- net expectancy in basis points per trade,
- LONG/SHORT trade counts and expectancy,
- target/stop/time exit rates,
- average holding time,
- profit factor,
- net EV by confidence bucket,
- fraction of walk-forward windows with positive net expectancy.

`--stress-execution` replays the exact same test signals under deterministic
adverse scenarios: doubled spread, doubled slippage, an extra bar of delay, and
a combined adverse case. These are diagnostics, not proof that those assumptions
match a particular broker or security.

Portfolio capital, correlation constraints, and risk-based sizing are intentionally
not mixed into V0.6; they belong to V0.7 after single-trade net expectancy survives
walk-forward and execution stress.

## V0.7 expected value + portfolio construction

V0.7 sits on top of the V0.6 event-driven execution layer. It does **not**
replace realistic fills with a classification shortcut.

The candidate pipeline is now:

```text
calibrated LONG / WAIT / SHORT probabilities
        ↓
structural net-EV prior after spread/slippage/fees
        ↓
earlier calibration/paper trade outcomes
        ↓
shrunken empirical EV by side + confidence bucket
        ↓
net EV / stop-risk ranking
        ↓
dynamic correlation penalty
        ↓
risk-based position sizing
        ↓
portfolio exposure constraints
        ↓
0..N selected positions (cash may remain idle)
```

The empirical EV model never learns from the same untouched test period it is
ranking. Sparse confidence buckets are shrunk toward the structural probability
prior so a handful of lucky trades cannot dominate sizing.

The allocator currently enforces:

- maximum simultaneous positions,
- maximum gross deployed capital,
- maximum single-position notional,
- fixed account risk per trade,
- maximum total account risk,
- maximum sector exposure,
- maximum LONG and SHORT gross exposure,
- directional-correlation cluster risk,
- minimum positive net EV,
- minimum viable position dollars.

Initial V0.7 sizing is deliberately **not Kelly sizing**. Quantity starts from:

```text
account risk budget / stop distance per share
```

and is only reduced by portfolio constraints. Kelly-style sizing should not be
tested until probability and EV calibration have survived real walk-forward
validation.

Run portfolio-level walk-forward research:

```bash
python -m research.portfolio_walk_forward \
  --dataset-version v05-r50-2y-001 \
  --model xgboost \
  --train-months 6 \
  --calibration-months 1 \
  --test-months 1 \
  --baseline-confidence 0.60 \
  --initial-equity 10000 \
  --spread-bps 4 \
  --slippage-bps 2 \
  --risk-per-trade 0.0025 \
  --max-total-risk 0.01 \
  --max-positions 3 \
  --min-net-ev-bps 0
```

For every walk-forward window the harness:

1. trains and calibrates the alpha model,
2. backtests the **earlier calibration period** to fit empirical EV,
3. estimates correlations from train+calibration rows only,
4. freezes both before the test boundary,
5. ranks untouched test signals by net EV,
6. constructs a constrained portfolio,
7. executes selected trades through V0.6, and
8. reports account return, max drawdown, realized trade EV, and the V0.6
   confidence-gated all-signal baseline side by side.

The V0.7 engineering layer being present is **not** evidence that V0.7 has
passed. The Master Plan pass condition still requires real multi-window data to
show improved portfolio economics without unacceptable drawdown.

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
