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

## V0.8 Laya specialist meta-policy

V0.8 turns Laya into a **trained veto/quality gate** after V0.7 ranking. It is
not allowed to create a trade, flip a direction, increase position size, or
bypass portfolio risk controls.

The production-intent pipeline is:

```text
V0.7 positive-net-EV candidates
        ↓
top 5 quantitative candidates only
        ↓
fine-tuned Laya specialist
        ↓
APPROVE or VETO
        ↓
unchanged V0.7 allocator + risk limits
```

The generic Laya checkpoint is **not** considered a V0.8 trading specialist.
The project first builds supervised historical examples from realized V0.6
execution outcomes. For every quant candidate the dataset builder simulates
both LONG and SHORT under the same spread/slippage/fee assumptions and derives
`LONG`, `SHORT`, or `WAIT` from what actually happened. The quant model's own
prediction is retained as input context but is never copied into the target.

Build the chronological specialist dataset:

```bash
python -m research.build_laya_dataset \
  --dataset-version v05-r50-2y-001 \
  --model xgboost \
  --train-months 6 \
  --calibration-months 1 \
  --test-months 1 \
  --spread-bps 4 \
  --slippage-bps 2 \
  --top-candidates 5 \
  --output-dir data/laya/v08
```

This produces:

```text
data/laya/v08/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

Each record contains a compact market/quant/portfolio state, the exact typed
Laya question schema, realized-outcome answers, and audit metadata. Splits are
chronological so the final test partition remains untouched.

Install the optional Laya runtime separately after selecting the appropriate
PyTorch build for the machine:

```bash
pip install -r requirements-laya.txt
```

Fine-tune a domain checkpoint using Laya's upstream supported fine-tuning
workflow, using `train.jsonl` for fitting and `validation.jsonl` for model
selection/calibration work. The final `test.jsonl` must stay untouched until
the checkpoint is frozen.

Run the frozen checkpoint on the validation split:

```bash
python -m research.laya_infer \
  --input data/laya/v08/validation.jsonl \
  --output data/laya/v08/validation_predictions.jsonl \
  --model YOUR_FINE_TUNED_LAYA_CHECKPOINT \
  --device cuda
```

Fit Joe Bot's held-out post-hoc temperatures. V0.8 gates on calibrated answer
probability rather than Laya's entropy-style `confidence` value:

```bash
python -m research.calibrate_laya \
  --predictions data/laya/v08/validation_predictions.jsonl \
  --output data/laya/v08/calibration.json
```

Finally compare the **same V0.7 portfolio** with and without Laya on held-out
walk-forward periods:

```bash
python -m research.laya_walk_forward \
  --dataset-version v05-r50-2y-001 \
  --laya-model YOUR_FINE_TUNED_LAYA_CHECKPOINT \
  --laya-calibration data/laya/v08/calibration.json \
  --device cuda
```

The report contains ungated and Laya-gated trade EV, average window return,
worst-window drawdown, veto counts, and deltas. V0.8 passes only when the
Laya-gated system improves held-out expectancy while drawdown degradation stays
inside the configured tolerance. If it does not improve the economics, Laya is
removed or its authority is reduced; it is not retained merely because it is
an AI model.

## V0.9 live paper engine

V0.9 adds the broker-facing execution/state foundation while keeping live capital
impossible. The Alpaca broker adapter in this phase is hard-wired to
`paper=True`; there is no configuration flag that can silently turn it into a
live-capital client.

The runtime now separates:

```text
Alpaca market WebSocket
        ↓
normalized bar/quote events + stream health
        ↓
validated candidate-provider interface
        ↓
independent production risk engine
        ↓
idempotent order manager
        ↓
Alpaca PAPER broker
        ↓
trade-update WebSocket
        ↓
local position state ↔ broker reconciliation
        ↓
exit engine
        ↓
append-only SQLite audit/replay
```

Implemented entry kill switches include stale market data, broker/trading-stream
disconnection, unstable market WebSocket, excessive clock drift, missing model,
invalid features, inconsistent position state, daily loss/drawdown, trade-count
and consecutive-loss limits, unexpected volatility, and broker trading blocks.
**Risk-reducing exits stay enabled while entry kill switches are active.**

Order submission uses stable client-order IDs and refuses a second broker submit
for an already-known intent. Broker order updates are validated against the
original symbol, side, quantity, filled quantity, and legal lifecycle
transitions. Position state is independently reconciled against broker truth;
an unexplained mismatch disables new entries.

Run the initial paper connectivity/state monitor:

```bash
python -m live.run_paper \
  --feed iex \
  --audit-db data/paper/audit.sqlite3
```

This runner intentionally has **new entries disabled**. It validates real-time
market/trade WebSockets, account and position reconciliation, existing paper
position exits, reconnect/health behavior, and the audit trail without letting
an unvalidated alpha stack generate orders. A validated V0.7/V0.8 live candidate
provider is plugged into the same `PaperTradingEngine` only after the earlier
empirical gates pass.

Replay one candidate/decision audit trail:

```bash
python -m live.replay DECISION_ID --audit-db data/paper/audit.sqlite3
```

V0.9 is not considered passed merely because this infrastructure exists. The
Master Plan pass condition still requires hundreds of correctly handled paper
trades across multiple sessions with no unexplained order, account, or position
state errors.

## V0.95 shadow production

V0.95 runs Joe Bot against **live market data with zero external orders**. It uses
an in-process `ShadowBroker` that has no broker SDK, credentials, trading endpoint,
or network submission method. Hypothetical market orders are acknowledged locally
and filled on the next eligible bar using the same adverse spread/slippage/tick
convention as V0.6; configured fees are also charged to virtual equity.

The shadow audit records:

- what Joe would have traded,
- decision timestamp and expected entry price,
- desired quantity, stop, target, and expected EV,
- next-bar hypothetical fill and fill slippage,
- signal-to-fill latency,
- realized virtual exit price/return/reason,
- market-stream health and kill switches.

Run shadow infrastructure in monitor-only mode:

```bash
python -m live.run_shadow --feed iex --audit-db data/shadow/audit.sqlite3
```

Plug in a validated real-time candidate stack with `module:function`:

```bash
python -m live.run_shadow \
  --provider your_runtime_module:build_candidates \
  --feed iex \
  --audit-db data/shadow/audit.sqlite3
```

The provider receives each closed `BarEvent` plus current engine state and returns
`EntryProposal` objects. This is the seam where the frozen V0.7 EV/portfolio stack
and optional V0.8 Laya veto are connected after their empirical gates pass.

Summarize expected-versus-realized shadow economics:

```bash
python -m live.shadow_report --audit-db data/shadow/audit.sqlite3
```

Shadow engineering being present does **not** mean the shadow phase has passed.
Promotion still requires a meaningful multi-session sample with clean audit/state
behavior and acceptable expected-versus-realized execution/economic drift.

### Built-in live intelligence provider

V0.95 can now run the frozen quantitative stack directly instead of requiring a
custom provider. The built-in path requires all pre-live artifacts explicitly;
it will not silently drop EV history or correlation controls:

```bash
python -m live.run_shadow \
  --model-bundle data/models/xgboost_v2_calibrated.pkl \
  --ev-trades data/runtime/ev_calibration_trades.parquet \
  --raw-version YOUR_IMMUTABLE_RAW_VERSION \
  --correlations data/runtime/correlations.csv \
  --feed iex
```

The live feature store is seeded from historical minute bars and uses the exact
Feature Engine V2 schema. It waits for synchronized same-minute SPY/QQQ plus the
configured universe before creating a market-context batch. Invalid/missing
features or model/runtime failures **fail closed** and produce no candidate.

Optional V0.8 Laya vetoing can be layered on the same provider with
`--laya-model` and `--laya-calibration`. Laya still cannot create trades, change
direction, or size positions; the V0.7 allocator remains authoritative.


## Victory Sprint 1 — governed data + artifacts

The consolidated `AI_Quick_Trader_Path_to_Victory.md` makes reproducibility the
first gate. Joe Bot now has an immutable artifact registry, dataset fingerprinting,
quality gates, fixed/ATR barrier sweeps, and a leakage-safe label tournament.

Run the full Sprint 1 pipeline on a machine with `requirements.txt` installed and
Alpaca data credentials available:

```bash
python -m research.victory_sprint1 \
  --version victory-r50-2y-001 \
  --years 2 \
  --feed iex
```

This will:

1. acquire the 50-stock research universe plus SPY/QQQ,
2. build the immutable raw + processed V2 lake,
3. hash every declared partition and freeze the dataset fingerprint,
4. enforce mandatory dataset gates including 50 training symbols and >=700 days,
5. run fixed-percentage and ATR-aware barrier sweeps, and
6. rank candidate label regimes using train+calibration data only.

The final test period is excluded from label-selection work. The resulting dataset
is not evidence of trading edge; it becomes the frozen input to Victory Sprint 2.

Artifact registry examples:

```bash
python -m research.artifacts register alpha-v001 alpha-model data/models/alpha.pkl \
  --dataset-version victory-r50-2y-001 \
  --dataset-fingerprint YOUR_FINGERPRINT

python -m research.artifacts verify alpha-v001
python -m research.artifacts promote alpha-v001 --channel shadow
```

Promotion fails when any declared file hash changes or any recorded gate fails.
