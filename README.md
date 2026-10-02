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

## Victory Sprint 2 — Core Alpha Tournament

Sprint 2 keeps the final outer test period out of feature/model ranking. The
selection chronology is nested:

```text
outer train
  ↓
inner fit → inner calibration
  ↓
outer calibration = selection holdout
  ↓
pre-final-test walk-forward
  ↓
freeze multiple challengers
  ↓
one-time final-test diagnostics (audit only)
```

Run it after Sprint 1 has produced and frozen the production dataset:

```bash
python -m research.victory_sprint2 \
  --dataset-version victory-r50-2y-001 \
  --finalists 2
```

Sprint 2 performs:

- V2 feature-group ablations for market benchmarks, breadth, relative strength,
  time-normalized activity, trend, volatility, and session/time features;
- XGBoost and HistGradientBoosting benchmarking, plus LightGBM/CatBoost when
  installed;
- sigmoid and isotonic probability calibration;
- pre-final-test chronological classification walk-forward diagnostics; and
- freezing of multiple governed alpha challengers for Sprint 3.

A challenger is deliberately registered with a failed
`sprint3_economic_proof` promotion gate. Classification quality alone can never
promote a model to shadow/paper use. Sprint 3 must prove net economic edge after
execution costs first.

## Victory Sprint 3 — Economic Proof / CORE EDGE

Sprint 3 is the first phase allowed to answer the question that matters: does a
frozen Sprint 2 challenger have positive, stable net economics after realistic
execution assumptions?

```bash
python -m research.victory_sprint3 \
  --artifact-id YOUR_SPRINT2_CHALLENGER_ID \
  --spread-bps 4 \
  --slippage-bps 2 \
  --entry-delay-bars 1
```

The campaign runs V0.6 event-driven walk-forward with adverse execution stress,
then V0.7 EV/portfolio walk-forward. The generated `core_edge_report.json`
contains an explicit `PASS`/`FAIL` gate covering positive net expectancy,
walk-forward stability, probability calibration, drawdown, cost-stress survival,
EV-vs-realized monotonicity, and single-symbol concentration.

If CORE EDGE fails, the proof artifact is registered as rejected and the roadmap
returns to labels/features/models. A passing CORE EDGE artifact only permits the
next research stage; it is **not** authorization for live-capital trading.

## Victory Sprint 4 — regime + specialist intelligence

Sprint 4 is deliberately gated behind Sprint 3. It refuses to run unless its
parent artifact is a verified `validated-alpha` with a CORE EDGE PASS.

The intelligence expansion layer adds:

- probabilistic regimes: `TREND_UP`, `TREND_DOWN`, `RANGE`, `HIGH_VOL`,
  `LOW_VOL`, `SHOCK`, `OPENING_VOLATILITY`,
- four transparent specialists: momentum, breakout, pullback, mean reversion,
- one shared specialist contract: `P(WAIT)`, `P(LONG)`, `P(SHORT)`, estimated
  EV and uncertainty,
- a calibrated multiclass meta-model trained on specialist/regime state,
- realized trade attribution by regime, and
- an `INTELLIGENCE EXPANSION` PASS/FAIL gate against the exact core baseline.

Run after a Sprint 3 validated-alpha exists:

```bash
python -m research.victory_sprint4 \
  --artifact-id YOUR_VALIDATED_ALPHA_ARTIFACT
```

Model/strategy selection remains pre-final-test. Sprint 4 passes only when the
meta-system improves net expectancy after identical costs, does not degrade the
positive-window rate, remains positive overall, and shows edge across more than
one regime. A failure keeps the simpler validated core alpha in control.

## Victory Sprint 5 — Laya + uncertainty + OOD

Sprint 5 keeps all secondary intelligence asymmetric. Laya, disagreement and
OOD may veto or reduce an already-approved quant trade; none may create a new
trade, flip its direction, or increase its risk.

The new uncertainty/OOD layer provides:

- normalized predictive entropy and cross-model directional disagreement,
- a risk multiplier that can fall to zero when disagreement is extreme,
- a robust feature-distribution OOD detector calibrated on an earlier partition,
- held-out diagnostics proving whether high uncertainty and OOD states are
  genuinely harder before those signals receive authority, and
- a governed Sprint 5 gate combining those diagnostics with the existing Laya
  held-out economic comparison.

A Sprint 5 component is not promoted merely because it sounds safer. Disagreement
must identify lower-quality states, OOD must identify degraded behavior, and Laya
must improve held-out expectancy without unacceptable drawdown degradation.

## Victory Sprint 6 — position + exit intelligence

Sprint 6 extends V0.6 trade telemetry with maximum favorable excursion (MFE) and
maximum adverse excursion (MAE), then measures alpha decay across holding bars.
It also provides a chronological continuation-vs-exit research model and a
strict `EXIT INTELLIGENCE` gate against the deterministic target/stop/time
baseline.

Adaptive exits remain subordinate to hard risk controls. They may exit early or
continue holding within the original risk envelope, but they may not widen a
hard stop, increase position size, bypass session flattening, or create a new
trade. If the adaptive policy does not improve held-out expectancy without
worsening drawdown, the deterministic exit engine remains authoritative.

## Victory Sprint 7 — microstructure + execution

Sprint 7 adds immutable historical quote/trade storage, causal minute-level
microstructure features, quote-aware partial/missed-fill simulation, and an
execution-policy benchmark. Market, marketable-limit, and passive-limit policies
can now be evaluated on both realized cost and fill quality.

The `EXECUTION POLICY` gate requires lower average execution cost, a high fill
rate, and no material deterioration in tail execution cost. Passing research is
still only a challenger: live shadow A/B measurement is required before it may
influence paper execution.

## Victory Sprint 8 — promotion factory

Sprint 8 packages passed research into one immutable runtime artifact containing
the frozen alpha bundle, empirical EV state, correlation matrix, risk config,
execution config, optional Laya calibration, feature drift baseline, hashes and
source-dataset lineage.

Runtime packages may be promoted only to `shadow` or `paper` at this stage.
Their manifest carries `live_capital_authorized: false`, and the Sprint 8
promotion API rejects `tiny-live` outright. Drift baselines use feature
quantiles and Population Stability Index (PSI) so shadow/paper runtime can
surface distribution shifts before they become unexplained trading behavior.

## Victory Sprint 9 — shadow + paper campaign proof

Sprint 9 turns V0.95/V0.9 infrastructure into an operational promotion gate. A
runtime package must first be explicitly promoted to both `shadow` and `paper`.
The campaign evaluator then scores real audit databases plus a deterministic
fault-injection suite.

Default proof thresholds are deliberately demanding and configurable:

- at least 5 shadow sessions,
- at least 100 closed shadow trades,
- positive realized shadow EV,
- absolute expected-vs-realized shadow EV drift <= 5 bps,
- at least 10 paper sessions,
- at least 300 completed paper trades,
- 100% clean-session rate,
- zero runtime crashes,
- zero unexplained order/position reconciliation mismatches,
- zero duplicate order submissions, and
- 100% pass rate on the local safety fault-injection campaign.

Run it with one or more append-only audit databases:

```bash
python -m research.victory_sprint9 \
  --runtime-package joe-runtime-v1 \
  --shadow-audit data/shadow/session-01.sqlite3 \
  --shadow-audit data/shadow/session-02.sqlite3 \
  --paper-audit data/paper/session-01.sqlite3 \
  --paper-audit data/paper/session-02.sqlite3 \
  --artifact-id joe-shadow-paper-proof-v1
```

The Sprint 9 artifact is either `shadow-paper-proof` or `campaign-failure`.
**Even a PASS keeps `live_capital_authorized=false`.** Tiny-live authorization is
a separate later gate and cannot be obtained from Sprint 9.

## Victory Sprint 10 — ARGUS Golden GUI / Operator OS

Sprint 10 adds a risk-first operator surface without changing Joe Bot's trading
authority. The backend is FastAPI and defaults to a read-only control adapter;
`live_capital_authorized` is always false at this stage.

Backend contract:

```text
REST
/api/command-center
/api/scanner
/api/portfolio
/api/positions
/api/risk
/api/system
/api/models
/api/laya
/api/backtests
/api/research/artifacts
/api/execution
/api/replay/{decision_id}

WebSocket
/ws/live/scanner
/ws/live/positions
/ws/live/orders
/ws/live/risk
/ws/live/system
```

Run the API after installing the optional dashboard dependencies:

```bash
pip install -r requirements-dashboard.txt
uvicorn argus_api.app:app --host 127.0.0.1 --port 8000
```

Run the Next.js dashboard:

```bash
cd dashboard
npm install
npm run dev
```

The dashboard currently includes Command Center, Live Scanner, Portfolio,
Positions, Model Console, Laya Console, Research Lab, Backtest Explorer,
Execution Console, Decision Replay, Risk Center, and System Health. The visual
language is deliberately restrained and risk-first; there are no live-capital
buttons in Sprint 10.

## Victory Sprint 11 — champion/challenger, dynamic universe, events

Sprint 11 implements the Golden Plan expansion layer without changing live-capital
authority. Challengers receive identical evidence and remain shadow-only until they
beat the champion across predefined EV, stress, calibration, drawdown, sample-size,
and walk-forward stability gates.

The dynamic universe engine selects only tradeable symbols after hard price,
liquidity, volume, and spread filters, then ranks eligible names using liquidity,
premarket/relative activity, movement, and structured event scores. Intraday
additions/removals are explicit and auditable.

Event/news intelligence is strongly typed and feature-only. External NLP/LLM output
must validate into `StructuredEvent`; it has no broker or order interface and cannot
directly create trades.

Example champion/challenger evaluation:

```bash
python -m research.victory_sprint11 \
  --champion data/experiments/champion.json \
  --challenger data/experiments/challenger.json
```

A Sprint 11 PASS means the challenger is eligible for further governed promotion;
`live_capital_authorized` remains false.

## Victory Sprint 12 — production hardening + formal go-live review

Sprint 12 adds production-hardening primitives without enabling live-capital trading.
The repository still has **no live-capital broker adapter** and the formal review
always reports `live_capital_authorized=false`.

Hardening now includes:

- atomic checksummed crash-recovery snapshots,
- service heartbeat monitoring and restart budgets,
- verified SQLite backup/restore,
- deterministic market-data source failover,
- availability/error/latency SLO evaluation,
- structured alert sinks,
- Git-tracked secret-hygiene scanning,
- least-privilege ARGUS RBAC primitives, and
- a formal go-live evidence review.

Run the formal review only after real upstream artifacts exist:

```bash
python -m research.victory_sprint12 \
  --artifact YOUR_VALIDATED_ALPHA \
  --artifact YOUR_RUNTIME_PACKAGE \
  --artifact YOUR_SHADOW_PAPER_PROOF \
  --recovery-test-passed \
  --supervisor-test-passed \
  --rbac-enabled \
  --backup-restore-test-passed
```

A passing review means only **eligible for a separately reviewed tiny-live
experiment**. It does not authorize live capital and it does not create an order
execution path.
