# Joe Bot — AI Quick Trader

Joe Bot is an experimental short-horizon US-equities quantitative research project. The repository is being upgraded from the original v0.3 dual-binary prototype according to `AI_Quick_Trader_Master_Plan.md`.

## Safety state

- Research / paper analysis only.
- No automatic live trading is enabled.
- The V0.4/Phase B research pipeline does not submit orders.
- Legacy broker scripts under `execution/` are not part of the offline unit-test suite.

## Current architecture

```text
Alpaca historical minute bars
        ↓
versioned immutable raw Parquet snapshot
        ↓
data-quality validation
        ↓
regular-session filtering
        ↓
session-safe features
        ↓
triple-barrier labels
        ↓
WAIT / LONG / SHORT multiclass XGBoost
        ↓
chronological purged split
        ↓
probability calibration
        ↓
out-of-sample confidence analysis
```

The legacy scanner remains available while the V0.4 model is researched and validated. It should not be treated as evidence of profitability.

## Setup

Create a virtual environment and install the research dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add **paper/data** Alpaca credentials. Never commit `.env`.

## Phase B — build the historical lake

The default research universe is a static, versionable 50-symbol starter universe. The original 17-symbol scanner universe remains unchanged for legacy compatibility.

Build two calendar years of one-minute regular-session history:

```bash
python -m research.build_dataset \
  --version v04-r50-2y-001 \
  --years 2 \
  --feed iex
```

Use `--feed sip` when your Alpaca data entitlement supports SIP and you explicitly want SIP history. The provider, feed, adjustment mode, date range, code commit, feature schema, barrier parameters, class distribution, quality statistics, and chronological split periods are frozen into the dataset manifest.

Data is stored as Parquet partitions:

```text
data/
  raw/
    datasets/<version>/bars/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/bars.parquet
  processed/
    datasets/<version>/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/training.parquet
  manifests/
    <version>.json
```

Dataset versions are immutable. Choose a new version name for every materially different raw snapshot or processed experiment.

Inspect a manifest without loading the entire dataset:

```bash
python -m research.report_dataset v04-r50-2y-001
```

Rebuild features/labels from an existing immutable raw snapshot without downloading history again:

```bash
python -m research.rebuild_dataset \
  --source-version v04-r50-2y-001 \
  --new-version v04-r50-2y-labels-002
```

Add `--reuse-source-label-config` when you want an exact label-configuration rebuild rather than applying the current label settings.

### Data-quality policy

The Phase B builder is deliberately conservative:

- corporate-action adjustment mode is explicit and versioned,
- current-symbol historical mapping uses Alpaca `asof`,
- timestamps are normalized to UTC,
- the initial research set is filtered to 09:30–16:00 America/New_York,
- duplicate timestamps are resolved deterministically,
- malformed/non-positive price rows and negative-volume rows are dropped and counted,
- impossible OHLC rows are dropped and counted,
- missing minute intervals are measured but **never forward-filled**,
- raw snapshots are immutable per dataset version.

## Train V0.4 from a versioned dataset

Training from the local lake is the preferred path:

```bash
python -m models.train_multiclass --dataset-version v04-r50-2y-001
```

The trainer reads the dataset's own manifest and uses its frozen label parameters and split fractions rather than silently applying newer settings.

A transient direct-download path still exists for debugging only:

```bash
python -m models.train_multiclass --download
```

## V0.4 model methodology

The V0.4 trainer:

- uses target-before-stop triple-barrier outcomes,
- maps ground truth to `WAIT=0`, `LONG=1`, `SHORT=2`,
- uses a single multiclass XGBoost model,
- balances classes during fitting,
- purges the label horizon around chronological validation boundaries,
- uses a later calibration holdout,
- evaluates only on the final untouched test period,
- reports confidence buckets so model confidence must prove useful out of sample.

## Offline tests

```bash
python -m unittest discover -s tests -v
```

The tests are broker-safe and do not place orders.
