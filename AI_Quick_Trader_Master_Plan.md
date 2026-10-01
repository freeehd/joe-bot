# AI Quick Trader — Master Development Plan

**Project status:** Prototype multi-stock scanner / ranker / allocator  
**Primary objective:** Build the smartest *feasibly possible* short-horizon US-equities trading system using quantitative models, Laya as a specialized decision layer, robust execution, and hard risk controls.  
**Core constraint:** The system must earn the right to use real money. No model, backtest, or AI component is assumed profitable until it proves itself out-of-sample, in paper trading, and then with very small live capital.

---

## 0. North-Star Goal

The end product should **not** be “an AI that guesses which stock goes up.”

It should be a complete trading system that continuously answers five different questions:

1. **Where is something worth looking at?**  
   Scan a large liquid universe and find unusual, tradeable activity.

2. **Is there a real statistical edge?**  
   Estimate LONG, SHORT, and WAIT probabilities from historical patterns.

3. **Is the trade worth taking after costs and risk?**  
   Estimate expected value after spread, slippage, latency, and risk.

4. **How much capital should be deployed?**  
   Allocate capital based on edge, confidence, volatility, correlation, and account-level limits.

5. **When should we get out?**  
   Re-evaluate every open position continuously and exit when the edge disappears, the target is reached, or the hard stop fires.

The finished architecture should look like:

```text
LIVE MARKET DATA
      │
      ▼
DYNAMIC UNIVERSE
      │
      ▼
FAST ACTIVITY SCANNER
      │
      ▼
FEATURE ENGINE
      │
      ├──────────────┐
      ▼              ▼
MARKET REGIME      COST / LIQUIDITY MODEL
      │              │
      └──────┬───────┘
             ▼
PRIMARY ALPHA MODEL
WAIT / LONG / SHORT
             │
             ▼
EXPECTED VALUE MODEL
             │
             ▼
TOP CANDIDATES ONLY
             │
             ▼
FINE-TUNED LAYA META-POLICY
             │
             ▼
PORTFOLIO / CAPITAL ALLOCATOR
             │
             ▼
HARD RISK ENGINE
             │
             ▼
EXECUTION ENGINE
             │
             ▼
BROKER
             │
             ▼
POSITION / EXIT MODEL
             │
             └──────────────→ repeat continuously
```

---

# 1. Where We Are Right Now

The current prototype already proves several important plumbing components.

## Current capabilities

- Python project running locally.
- CUDA/PyTorch confirmed on the RTX 3070.
- Laya loads and performs typed decisions.
- Alpaca historical market data is accessible.
- Feature generation works.
- XGBoost training works.
- Multi-stock scanning works.
- Separate LONG and SHORT XGBoost models exist.
- Stocks are ranked by a composite opportunity score.
- A capital allocator can split a predefined amount across candidates.
- Cash can remain unused.
- No automatic live trading is enabled.

Current high-level flow:

```text
~17 stocks
   ↓
5-day market data
   ↓
basic features
   ↓
LONG XGBoost + SHORT XGBoost
   ↓
activity filter
   ↓
ranker
   ↓
capital allocator
```

## Current known weaknesses

These are not minor polish issues; they are the core reasons the present version should **not** trade real money.

1. LONG and SHORT models are independent binary models and can both produce high probabilities.
2. Current labels are still based on simple future returns rather than realistic trade outcomes.
3. The feature set is intentionally basic.
4. Relative volume is not time-of-day normalized.
5. Market context is weak.
6. There is no true net expected-value model.
7. There is no quote-aware slippage / spread model.
8. There is no rigorous event-driven backtester yet.
9. There is no walk-forward validation framework.
10. Laya is still generic / zero-shot rather than trained specifically on trading states.
11. There is no dedicated exit model.
12. There is no production-grade order state machine.
13. There is no full portfolio correlation / sector exposure model.
14. There is no live monitoring / kill-switch service.
15. There is no evidence yet that the strategy has positive out-of-sample expectancy.

---

# 2. Project Philosophy

The final system should be optimized for:

- **Selective trading**, not frequent trading.
- **Net expected value**, not raw prediction accuracy.
- **Capital preservation**, not maximum deployment.
- **Realistic execution**, not perfect backtest fills.
- **Calibrated probabilities**, not impressive-looking confidence scores.
- **Multiple independent evidence sources**, not one model controlling everything.
- **WAIT / CASH** as a first-class decision.
- **Continuous exits**, not “enter and hope.”

The bot must be able to say:

```text
There are no sufficiently good trades right now.
Deploy $0.
```

That is an intelligent action.

---

# 3. Target Trading Style

## Initial target

Build for **short-horizon intraday trading**, not exchange-level HFT.

Recommended first operating envelope:

| Property | Initial target |
|---|---|
| Asset class | Liquid US equities |
| Session | Regular trading hours |
| Decision horizon | ~30 sec to 5 min |
| Typical holding time | ~2–20 min |
| Data granularity | 1-min bars + live quotes/trades |
| Initial universe | 50–200 liquid symbols |
| Mature universe | 300–1000+ screened liquid symbols |
| Simultaneous positions | 0–3 initially |
| Primary objective | Net positive EV after trading costs |

True microsecond HFT is out of scope because competing with colocated market makers is not feasible from a retail setup.

---

# 4. PHASE A — Fix the Learning Target

**Priority: CRITICAL**

The current “price > +0.2% five minutes later” label is too crude.

## A1. Implement target-before-stop labels

For every historical timestamp, simulate a potential trade.

Example LONG:

```text
Entry:     100.00
Target:    100.30  (+0.30%)
Stop:       99.85  (-0.15%)
Horizon:    10 minutes
```

Inspect future bars in chronological order.

Possible outcomes:

```text
LONG_WIN
LONG_LOSS
NO_RESOLUTION
```

Repeat symmetrically for SHORT.

## A2. Move toward triple-barrier labeling

Each hypothetical trade has three barriers:

1. Profit target.
2. Stop loss.
3. Time expiration.

This is much closer to the real decision the bot must make.

## A3. Create one multiclass ground-truth label

Target:

```text
0 = WAIT
1 = LONG
2 = SHORT
```

Example logic:

```text
LONG target hits before long stop
AND short setup does not dominate
→ LONG

SHORT target hits before short stop
AND long setup does not dominate
→ SHORT

Neither / ambiguous / both / insufficient EV
→ WAIT
```

## A4. Make targets volatility-aware

Do not permanently use fixed percentages.

Eventually define:

```text
target_distance = ATR × target_multiplier
stop_distance   = ATR × stop_multiplier
```

This prevents treating a 0.3% move in a low-volatility stock as equivalent to a 0.3% move in a high-volatility stock.

### Exit gate for Phase A

Do not proceed until:

- Labels can be generated reproducibly.
- LONG/SHORT/WAIT frequencies are understood.
- No future information leaks into feature calculation.
- Target, stop, and time barrier logic has unit tests.

---

# 5. PHASE B — Build a Serious Historical Dataset

**Priority: CRITICAL**

A model is only as good as the history it learns from.

## B1. Expand the universe

Progression:

```text
17 symbols
→ 50
→ 100
→ 300
→ 500+
```

Prefer highly liquid US equities.

Avoid initially:

- extremely low-priced stocks,
- thinly traded names,
- huge spreads,
- frequent trading halts,
- symbols with poor quote quality.

## B2. Expand historical coverage

Do not train on 30–60 days only.

Target:

```text
Minimum useful research set:
1–2 years

Better:
3–5 years if data quality allows
```

The dataset should contain different regimes:

- bull markets,
- selloffs,
- high-volatility periods,
- low-volatility periods,
- earnings seasons,
- rate-shock days,
- quiet sessions,
- trend days,
- range days.

## B3. Store data locally

Use Parquet rather than repeatedly downloading everything.

Suggested layout:

```text
data/
  raw/
    bars/
    quotes/
    trades/
  processed/
    features/
    labels/
  models/
  backtests/
```

Partition by:

```text
symbol / year / month
```

## B4. Create a dataset manifest

For every training run store:

```text
dataset version
date range
symbols
features
label parameters
target multiplier
stop multiplier
time horizon
market-data source
row count
class distribution
```

This becomes essential once dozens of experiments exist.

### Exit gate for Phase B

- Historical dataset is versioned.
- Data can be rebuilt from raw inputs.
- Missing-data handling is explicit.
- Corporate-action handling is correct.
- Train/validation/test periods are defined chronologically.

---

# 6. PHASE C — Feature Engine V2

The current seven features prove the system works but are not enough for an exceptional trader.

## C1. Price / momentum features

Add:

```text
1m return
2m return
3m return
5m return
10m return
15m return

return acceleration
rolling momentum
distance from rolling high
distance from rolling low
breakout strength
pullback depth
```

## C2. Trend features

```text
EMA 5 / 10 / 20 / 50
EMA slopes
EMA separation
price-to-EMA distances
trend persistence
VWAP distance
VWAP slope
VWAP cross age
```

## C3. Volatility features

```text
ATR
normalized ATR
1m realized volatility
5m realized volatility
15m realized volatility
range expansion
Bollinger width
volatility acceleration
```

## C4. Volume features

Replace the primitive rolling-volume comparison.

Use:

```text
current volume
volume acceleration
cumulative session volume
volume / expected volume for this exact time of day
trade count
trade-count acceleration
```

### Time-of-day relative volume

Instead of:

```text
10:03 volume
vs
previous 20 minutes
```

compare:

```text
today 10:03 volume
vs
historical 10:03 volume
```

This corrects the distorted RVOL values near the open and close.

## C5. Market context

Every stock should know:

```text
SPY 1m / 5m / 15m
QQQ 1m / 5m / 15m
sector ETF returns
VIX / volatility context where available
stock relative strength vs SPY
stock relative strength vs QQQ
breadth / percentage of universe rising
```

## C6. Time features

```text
minutes since open
minutes until close
opening 15m flag
opening hour flag
lunch period flag
power-hour flag
day of week
```

## C7. Quote / microstructure features

When live quote data is available:

```text
bid
ask
spread
spread %
spread change
bid size
ask size
bid/ask imbalance
quote update rate
trade rate
short-term order-flow imbalance
```

## C8. Feature discipline

Do not dump hundreds of indicators into the model.

Every feature should survive:

- leakage tests,
- stability tests,
- feature-importance analysis,
- out-of-sample ablation.

Target a strong set of approximately **30–80 meaningful features** before considering massive feature expansion.

---

# 7. PHASE D — Primary Alpha Model V2

Replace separate LONG/SHORT binary models with a single multiclass model.

## D1. Model output

```text
P(WAIT)
P(LONG)
P(SHORT)
```

Example:

```text
MU

WAIT      54%
LONG      33%
SHORT     13%
```

Much cleaner than:

```text
LONG 40%
SHORT 38%
```

## D2. Candidate models

Benchmark several model families:

- XGBoost
- LightGBM
- CatBoost
- HistGradientBoosting
- shallow neural network only if justified

Do **not** assume XGBoost must win.

The production model is whichever performs best out-of-sample after costs.

## D3. Optimize for the right metrics

Generic accuracy is insufficient.

Track:

```text
precision LONG
precision SHORT
recall LONG
recall SHORT
macro F1
ROC-AUC where meaningful
PR-AUC
log loss
Brier score
calibration error
```

Most importantly:

```text
net EV per model-confidence bucket
```

Example:

```text
Confidence      Net avg trade
50–55%          -0.03%
55–60%          +0.01%
60–65%          +0.07%
65–70%          +0.13%
70%+            +0.20%
```

That is how the trading threshold should be chosen.

## D4. Calibrate probabilities

Use calibration on validation data:

- isotonic calibration,
- Platt / logistic calibration,
- or another validated method.

The bot needs:

```text
70%
```

to mean something close to:

```text
historically about 70% under comparable labeling conditions
```

not merely “the model emitted 0.70.”

---

# 8. PHASE E — Expected-Value Model

This is where the system becomes much smarter than “pick highest probability.”

For each candidate estimate:

```text
probability of target
probability of stop
expected favorable excursion
expected adverse excursion
expected holding time
expected spread cost
expected slippage
```

Then compute:

```text
Gross EV
=
P(win) × expected_win
-
P(loss) × expected_loss
```

Then:

```text
Net EV
=
Gross EV
-
spread
-
slippage
-
fees
-
expected execution penalty
```

Only trade when:

```text
Net EV > minimum required edge
```

This should eventually replace arbitrary probability thresholds.

---

# 9. PHASE F — Regime Intelligence

A strategy that works in one market regime often fails in another.

Create a regime classifier.

Possible regimes:

```text
TREND_UP
TREND_DOWN
RANGE
HIGH_VOL
LOW_VOL
PANIC / SHOCK
OPENING_VOLATILITY
```

Inputs can include:

- SPY / QQQ momentum,
- volatility,
- breadth,
- intraday range,
- volume,
- cross-sectional dispersion.

The candidate model receives the regime as a feature.

Later the system can learn different thresholds per regime.

Example:

```text
TREND_UP
→ momentum continuation gets more weight

RANGE
→ breakout signals require stronger confirmation

SHOCK
→ reduce size or disable trading
```

---

# 10. PHASE G — Laya Becomes a Trading Specialist

Laya should **not** merely duplicate XGBoost.

It becomes a **meta-policy / decision layer**.

## G1. Build a Laya training dataset from market history

Each example should contain:

```text
symbol
time
market regime
price features
volume features
volatility features
SPY context
QQQ context
sector context
spread
liquidity
primary model probabilities
expected-value estimate
current portfolio state
```

Target:

```text
LONG
SHORT
WAIT
```

The label must come from **realized historical outcomes / backtester ground truth**, not from whatever XGBoost predicted.

## G2. Critical rule

Bad:

```text
XGBoost says LONG
→ train Laya label = LONG
```

That just teaches Laya to imitate XGBoost.

Correct:

```text
XGBoost says LONG = 73%
+
all contextual features
+
actual trade outcome says WAIT was optimal

→ Laya label = WAIT
```

Now Laya can learn when XGBoost tends to be wrong.

## G3. Fine-tune Laya

Use Laya's supported fine-tuning workflow.

Training should include:

```text
train split
validation split
held-out test split
confidence calibration
```

Laya should only see the held-out period after training is completely frozen.

## G4. Role in production

Do **not** run Laya across 500 symbols.

Use:

```text
500 symbols
   ↓
cheap scanner
   ↓
50 candidates
   ↓
quant model
   ↓
5 strongest
   ↓
Laya
```

Laya then answers:

```text
LONG
SHORT
WAIT
setup quality
risk concern
```

## G5. Laya veto power, not unlimited authority

Initially:

```text
quant says trade
+
Laya says WAIT
→ reject

quant says WAIT
+
Laya says trade
→ still reject
```

This makes Laya a contextual safety / quality gate first.

Only after extensive evaluation should it gain more policy authority.

---

# 11. PHASE H — News / Event Intelligence

This is **not** part of the latency-critical first version.

Eventually add a small LLM or specialized NLP model to classify:

```text
earnings
guidance
analyst changes
SEC filings
mergers
product announcements
legal events
macro headlines
```

Output structured features:

```text
event_type
sentiment
surprise
expected_impact
time_horizon
credibility
```

Feed these into the decision system.

The LLM should not directly place trades.

---

# 12. PHASE I — Opportunity Scanner V2

The mature scanner should cover hundreds of symbols efficiently.

## Stage 1 — ultra-cheap screen

Filter using:

```text
liquidity
spread
price range
volume
recent movement
session activity
```

Example:

```text
700 symbols
→ 120 active
```

## Stage 2 — feature + alpha model

```text
120
→ 15 statistically interesting
```

## Stage 3 — cost/EV filter

```text
15
→ 7 with positive estimated net EV
```

## Stage 4 — Laya meta-policy

```text
7
→ 2–4 approved candidates
```

## Stage 5 — portfolio allocator

```text
2–4
→ 0–3 actual positions
```

---

# 13. PHASE J — Portfolio & Capital Allocation

The allocator should optimize **portfolio-level expected return per unit of risk**, not simply split money.

## Inputs

For every candidate:

```text
net expected value
calibrated probability
stop distance
volatility
liquidity
spread
sector
correlation to current positions
regime
Laya decision
```

## Constraints

```text
max total deployed capital
max position size
max risk per trade
max total account risk
max correlated exposure
max sector exposure
max LONG exposure
max SHORT exposure
max number of positions
```

## Position sizing

Initial production sizing should be risk-based:

```text
risk_budget_dollars
/
stop_distance_dollars
=
share quantity
```

Later, only after calibration is proven, test **fractional Kelly-style sizing** with a severe cap.

Never allow uncapped Kelly sizing.

## Capital can remain idle

Examples:

```text
$500 available
1 strong trade
→ deploy $80
→ keep $420 cash
```

or:

```text
No positive net EV
→ deploy $0
```

---

# 14. PHASE K — Correlation & Exposure Intelligence

Three stocks are not necessarily three independent bets.

Example:

```text
NVDA
AMD
AVGO
```

can represent one concentrated semiconductor factor.

Build:

```text
rolling correlation matrix
sector classification
factor exposure
market beta
```

The allocator should penalize highly correlated simultaneous positions.

---

# 15. PHASE L — Exit Intelligence

Entries get attention; exits often determine whether a strategy survives.

Build a **separate exit model**.

Inputs:

```text
entry price
current price
time in position
current unrealized PnL
maximum favorable excursion
maximum adverse excursion
updated alpha probabilities
updated EV
volume deterioration
spread deterioration
market regime change
SPY/QQQ reversal
```

Outputs:

```text
HOLD
EXIT
TAKE_PROFIT
REDUCE
```

The model should be re-evaluated continuously.

## Hard stops remain outside AI

If the hard stop fires:

```text
EXIT
```

No model may override it.

---

# 16. PHASE M — Execution Engine

Quick trading can lose its entire statistical edge through poor fills.

## M1. Use streaming data

Move from repeated historical polling to WebSocket market data.

Stream:

```text
trades
quotes
bars
```

Maintain an in-memory state per symbol.

## M2. Build an order state machine

Every order must have a lifecycle:

```text
CREATED
SUBMITTED
ACKNOWLEDGED
PARTIALLY_FILLED
FILLED
CANCEL_REQUESTED
CANCELLED
REJECTED
```

Never assume submit = fill.

## M3. Handle

- partial fills,
- rejected orders,
- stale orders,
- reconnects,
- duplicate messages,
- API timeouts,
- cancelled orders,
- market closure,
- trading halts.

## M4. Execution policy

Benchmark:

```text
market
limit
marketable limit
stop
stop-limit
bracket
trailing stop
```

For fast entries, a marketable limit policy may often be worth testing because it caps terrible fills while remaining executable.

Execution policy must be tested, not guessed.

---

# 17. PHASE N — Realistic Backtesting Engine

This is one of the most important pieces in the entire project.

The backtester must execute the **same strategy interfaces as live trading**.

```text
historical event
→ feature engine
→ models
→ Laya
→ allocator
→ risk
→ simulated execution
→ position manager
```

## Include realistic assumptions

At minimum:

```text
spread
slippage
latency
trade delay
partial-fill approximation
fees
minimum tick
session boundaries
corporate actions
```

Never assume entry at the exact candle close that generated the signal unless that is genuinely executable.

---

# 18. PHASE O — Walk-Forward Validation

Never judge the strategy from one train/test split.

Use rolling evaluation.

Example:

```text
Train: Jan–Jun
Validate: Jul
Test: Aug

Train: Feb–Jul
Validate: Aug
Test: Sep

Train: Mar–Aug
Validate: Sep
Test: Oct
```

Track whether edge survives across windows.

## Advanced validation

Once the basic framework is stable, add:

- purged time-series validation,
- embargo periods,
- parameter stability analysis,
- regime-stratified testing.

---

# 19. PHASE P — Stress Testing

Before paper execution, intentionally make assumptions worse.

Test:

```text
2× expected spread
2× slippage
100–300ms delay
missed fills
random order failures
WebSocket disconnects
market-data staleness
sudden volatility spikes
```

A strategy that only works under perfect execution is not a robust strategy.

---

# 20. PHASE Q — Metrics That Actually Matter

Track at least:

## Returns

```text
net PnL
return on deployed capital
return on account capital
average trade EV
median trade
```

## Risk

```text
max drawdown
daily drawdown
CVaR / tail loss
largest loss
consecutive losses
```

## Trade quality

```text
profit factor
average winner
average loser
win rate
expectancy
MFE
MAE
holding time
```

## Model quality

```text
calibration
Brier score
precision by confidence bucket
net PnL by confidence bucket
performance by regime
performance by symbol
```

## Execution

```text
decision-to-order latency
order-to-fill latency
slippage
spread paid
fill rate
cancel rate
reject rate
```

---

# 21. PHASE R — Paper Trading

Only after backtesting survives the earlier gates.

Paper system must run exactly like the intended live system.

## Minimum paper-trading objectives

Test:

- live streaming,
- decisions,
- order lifecycle,
- exits,
- reconnects,
- market open/close,
- logging,
- kill switches,
- portfolio limits.

Paper trading is **execution validation**, not proof of profitability.

## Minimum sample

Do not evaluate from 10–20 trades.

Target:

```text
hundreds of trades
across many sessions
and multiple regimes
```

Preferably enough data to compare:

```text
expected EV
vs
realized paper EV
```

---

# 22. PHASE S — Shadow Mode

Before allowing live orders:

Run against the live market but place **no orders**.

Record:

```text
what would have been bought
when
at what expected price
desired quantity
predicted target
predicted stop
what actually happened
```

This is valuable for finding:

- timing issues,
- data latency,
- feature mismatches,
- model drift,
- unrealistic backtest assumptions.

---

# 23. PHASE T — Tiny Live Capital

Only after:

```text
backtest PASS
walk-forward PASS
stress tests PASS
paper PASS
shadow PASS
```

move to tiny live capital.

Initial live objective:

> Verify execution realism — not maximize profit.

Example progression:

```text
Stage 1:
minimal capital

Stage 2:
small risk budget

Stage 3:
gradual scaling after evidence
```

Never jump from profitable paper trading directly to meaningful capital.

---

# 24. PHASE U — Production Risk Engine

Risk control must be logically independent from Laya and the alpha models.

## Per-trade limits

```text
max dollar risk
max % account risk
max position notional
max spread
min liquidity
max volatility
```

## Portfolio limits

```text
max positions
max gross exposure
max net exposure
max sector exposure
max correlated exposure
```

## Daily limits

```text
max daily loss
max daily drawdown
max number of trades
max consecutive losses
```

## System kill switches

Disable new trading if:

```text
market data stale
broker disconnected
clock drift detected
WebSocket unstable
model missing/corrupt
feature values invalid
position state inconsistent
daily loss breached
unexpected volatility
```

Exits must still remain possible while new entries are disabled.

---

# 25. PHASE V — Full Audit Log / Replay System

Every decision must be reproducible.

For each candidate store:

```text
timestamp
symbol
raw market snapshot
feature vector
regime
model version
P(WAIT)
P(LONG)
P(SHORT)
EV estimate
Laya input
Laya output
allocator state
risk decision
order decision
actual order
fill
exit
PnL
MFE
MAE
```

Then support:

```text
replay decision #12345
```

and reconstruct exactly why the bot acted.

This becomes the dataset for future improvement.

---

# 26. PHASE W — Dashboard

Build a Next.js dashboard only after the core intelligence is trustworthy.

Views:

## Live scanner

```text
Ticker
Direction
P(LONG)
P(SHORT)
P(WAIT)
Net EV
Regime
Spread
Rank
```

## Portfolio

```text
open positions
entry
current
PnL
stop
target
model edge now
exit recommendation
```

## Risk

```text
daily PnL
daily drawdown
gross exposure
sector exposure
risk budget remaining
kill-switch state
```

## Model health

```text
current model version
calibration drift
recent hit rate
recent EV
feature drift
```

---

# 27. PHASE X — Drift Detection & Retraining

Markets change.

Monitor:

```text
feature distribution drift
prediction distribution drift
calibration drift
EV deterioration
regime-specific degradation
```

Retraining should be controlled.

Do not:

```text
lose money today
→ automatically retrain tonight
```

Instead:

```text
new candidate model
→ offline test
→ walk-forward
→ paper/shadow comparison
→ promote only if superior
```

Maintain a model registry:

```text
champion
challenger
retired
```

---

# 28. PHASE Y — Champion / Challenger Architecture

Run multiple strategies in parallel without all receiving money.

Example:

```text
Champion:
production model

Challenger A:
new features

Challenger B:
new label horizon

Challenger C:
new Laya checkpoint
```

All receive the same market stream.

Only champion trades.

Compare realized hypothetical outcomes.

This makes improvement continuous without destabilizing production.

---

# 29. PHASE Z — Advanced Strategy Ensemble

Once the base system is proven, split alpha generation into specialists.

Examples:

```text
Momentum breakout model
Pullback continuation model
Mean-reversion model
Opening-range model
Volatility expansion model
```

A meta-model selects which strategy is appropriate for the current regime.

Architecture:

```text
Market Regime
    │
    ├── Momentum model
    ├── Pullback model
    ├── Mean-reversion model
    └── Breakout model
             │
             ▼
        Meta-policy
             │
             ▼
            Laya
```

Do not build this before the first strategy has demonstrated real edge.

---

# 30. Deployment Architecture

## Development

Your current machine:

```text
Ryzen 7 5800X
RTX 3070
```

is excellent for:

- local development,
- XGBoost training,
- feature research,
- backtesting,
- Laya inference.

## Laya fine-tuning

The official Laya project provides a fine-tuning workflow designed around 2× T4 GPUs.

Use:

- cloud / Kaggle style training where appropriate,
- local RTX 3070 mainly for inference and smaller experiments.

## Serious live trading deployment

For US equities, eventually move the production execution system close to the broker/data infrastructure.

Recommended:

```text
US-East VPS / cloud instance
```

Run:

```text
market stream
feature engine
XGBoost
risk engine
execution engine
database
```

there.

Laya can either:

1. also run on the server if latency is acceptable, or
2. remain outside the absolute critical path.

Your home RTX 3070 is ideal for development, but a home connection from Pakistan should not become a single point of failure for a production short-horizon trading engine.

---

# 31. Proposed Final Technology Stack

## Core

```text
Python 3.11
```

## Data / computation

```text
Polars
NumPy
Pandas where needed
PyArrow / Parquet
```

## Models

```text
XGBoost
LightGBM
CatBoost
scikit-learn calibration
Laya
PyTorch
```

## Research

```text
Jupyter
Matplotlib
Optuna
MLflow or lightweight equivalent
```

## Storage

Development:

```text
Parquet
SQLite
```

Production:

```text
PostgreSQL
Parquet historical lake
Redis optional for live state
```

## Broker / data

```text
Alpaca initially
```

Keep broker abstraction so the system can later support:

```text
Interactive Brokers
other market-data vendors
```

without rewriting strategy logic.

## Dashboard

```text
Next.js
```

## Production

```text
Docker
systemd / container orchestration
structured logging
health checks
```

---

# 32. Recommended Project Structure

```text
ai-trader/
│
├── config/
│   ├── settings.py
│   ├── risk.py
│   └── model_versions.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── labels/
│   ├── models/
│   └── backtests/
│
├── market/
│   ├── historical.py
│   ├── stream.py
│   ├── snapshots.py
│   ├── universe.py
│   └── scanner.py
│
├── features/
│   ├── price.py
│   ├── volume.py
│   ├── volatility.py
│   ├── microstructure.py
│   ├── market_context.py
│   └── engine.py
│
├── labels/
│   ├── triple_barrier.py
│   └── outcomes.py
│
├── models/
│   ├── alpha/
│   ├── regime/
│   ├── execution/
│   ├── exit/
│   └── laya/
│
├── strategy/
│   ├── ranker.py
│   ├── expected_value.py
│   ├── allocator.py
│   └── policy.py
│
├── risk/
│   ├── trade_risk.py
│   ├── portfolio_risk.py
│   └── kill_switch.py
│
├── execution/
│   ├── broker.py
│   ├── order_manager.py
│   ├── fills.py
│   └── position_manager.py
│
├── backtest/
│   ├── engine.py
│   ├── fills.py
│   ├── walk_forward.py
│   └── metrics.py
│
├── monitoring/
│   ├── logger.py
│   ├── drift.py
│   └── alerts.py
│
├── dashboard/
│
├── tests/
│
└── main.py
```

---

# 33. Development Milestones

## V0.4 — Correct Intelligence Foundation

Build:

- target-before-stop labels,
- LONG/SHORT/WAIT multiclass XGBoost,
- multi-stock historical dataset,
- directional calibration.

**No trading.**

### Pass condition

Model demonstrates meaningful out-of-sample separation between confidence buckets.

---

## V0.5 — Market-Aware Model

Add:

- SPY,
- QQQ,
- sector context,
- proper relative volume,
- volatility,
- session features.

### Pass condition

New model improves held-out EV, not merely training metrics.

---

## V0.6 — Backtester

Build event-driven simulation.

Add:

- spread,
- slippage,
- latency,
- realistic signal timing.

### Pass condition

Positive out-of-sample net expectancy under conservative cost assumptions.

---

## V0.7 — EV + Ranking + Portfolio

Build:

- expected-value model,
- correlation-aware allocator,
- risk-based sizing.

### Pass condition

Portfolio-level walk-forward metrics improve versus single-trade baseline without unacceptable drawdown.

---

## V0.8 — Laya Specialist

Build:

- historical Laya dataset,
- fine-tune,
- calibration,
- top-candidate Laya gate.

### Pass condition

Adding Laya improves held-out net EV / drawdown versus identical system without Laya.

If it does not improve results, do not keep it merely because it is AI.

---

## V0.9 — Live Paper Engine

Build:

- WebSockets,
- order manager,
- live position manager,
- exit engine,
- kill switches.

### Pass condition

Hundreds of paper trades execute correctly with no unexplained position/account-state errors.

---

## V0.95 — Shadow Production

Live data.

No trades.

Compare:

```text
predicted fills
vs
observable market behavior
```

### Pass condition

Real-time assumptions remain consistent with research results.

---

## V1.0 — Tiny Live

Small capital only.

### Pass condition

Live execution quality and realized results remain compatible with expected ranges.

---

## V1.1+ — Scaling

Only scale after sufficient live evidence.

Scale:

```text
slowly
in predefined steps
```

Do not automatically increase size after a few wins.

---

# 34. Hard Go / No-Go Criteria Before Real Money

The exact thresholds should be selected from research rather than invented now, but the following categories are mandatory.

The system must demonstrate:

### Statistical

- positive out-of-sample expectancy,
- stable performance across multiple walk-forward windows,
- useful probability calibration,
- no single-symbol dependency.

### Economic

- profitable after conservative costs,
- acceptable drawdown,
- sensible profit factor,
- edge survives increased slippage.

### Operational

- no duplicate orders,
- correct recovery after disconnect,
- correct partial-fill handling,
- position state reconciles with broker,
- kill switch tested.

### Paper

- substantial sample size,
- multiple market regimes,
- paper execution behaves close enough to modeled assumptions.

### Live

Begin only with capital whose loss is acceptable.

---

# 35. Things We Should Explicitly Avoid

Do not:

- optimize only for win rate,
- train and test on randomly shuffled market rows,
- let Laya directly submit orders,
- let an LLM generate arbitrary order parameters,
- force capital deployment,
- assume paper fills equal live fills,
- optimize repeatedly against the final test set,
- add dozens of indicators simply to make the system look sophisticated,
- use uncalibrated confidence as probability,
- scale because of one good week,
- let the system average down without predefined rules,
- remove stops because “the model still believes,”
- retrain automatically after every losing period.

---

# 36. What “Exceptional” Should Mean

An exceptional bot is **not** the one with the fanciest AI diagram.

It should demonstrate:

```text
better opportunity selection
fewer low-quality trades
lower execution waste
well-calibrated confidence
positive net expectancy
controlled drawdowns
fast failure detection
automatic abstention
robust operation
full auditability
```

The strongest version of this system may trade **less** than a mediocre version.

That is acceptable.

---

# 37. Immediate Next Build Order

From the current codebase, work in this exact order:

```text
1. Freeze current V0.3 code in Git.

2. Create target-before-stop / triple-barrier labeling.

3. Replace dual binary XGBoost with:
   WAIT / LONG / SHORT multiclass XGBoost.

4. Train across the full multi-stock historical dataset.

5. Add proper chronological validation.

6. Add SPY + QQQ + sector market context.

7. Replace current RVOL with time-of-day normalized RVOL.

8. Build probability calibration.

9. Build an event-driven backtester.

10. Add realistic spread/slippage/latency.

11. Measure net EV by confidence bucket.

12. Build expected-value ranking.

13. Upgrade capital allocator to risk/correlation aware.

14. Build Laya training dataset from historical outcomes.

15. Fine-tune + calibrate Laya.

16. Prove Laya improves held-out economics.

17. Build separate exit model.

18. Build WebSocket live state engine.

19. Build robust order/position state machine.

20. Add production risk engine + kill switches.

21. Paper trade hundreds of trades.

22. Run shadow mode.

23. Tiny live deployment.

24. Champion/challenger continuous improvement.

25. Scale only when live evidence supports it.
```

---

# 38. Final Target Architecture

```text
                     US MARKET DATA
                           │
                  ┌────────┴────────┐
                  │                 │
             HISTORICAL           LIVE
                  │                 │
                  └────────┬────────┘
                           ▼
                  DATA QUALITY LAYER
                           │
                           ▼
                   DYNAMIC UNIVERSE
                           │
                           ▼
                  ACTIVITY SCANNER
                           │
                           ▼
                     FEATURE ENGINE
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
 MARKET REGIME       MICROSTRUCTURE       MARKET CONTEXT
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                MULTICLASS ALPHA MODEL
                WAIT / LONG / SHORT
                           │
                           ▼
                    CALIBRATION
                           │
                           ▼
                  EXPECTED VALUE MODEL
                           │
                           ▼
                  TOP CANDIDATES ONLY
                           │
                           ▼
                 FINE-TUNED LAYA POLICY
                           │
                           ▼
                   PORTFOLIO ALLOCATOR
                           │
                           ▼
                     HARD RISK ENGINE
                           │
                           ▼
                    EXECUTION ENGINE
                           │
                           ▼
                        BROKER
                           │
                           ▼
                    OPEN POSITIONS
                           │
                           ▼
                      EXIT MODEL
                           │
                           └───────────┐
                                       ▼
                              CONTINUOUS LOOP
```

---

# 39. Reality Check

The project can be engineered to be **far smarter, more selective, more measurable, and more robust** than the current prototype.

What cannot be engineered is a guarantee of profit.

The actual objective is therefore:

> Build the strongest feasible short-horizon trading research and execution system, then require evidence that it possesses positive net expectancy before allowing it to control meaningful capital.

If the evidence says a component does not improve results — including Laya, a feature, a strategy, or an allocator — remove it.

That discipline is what gives the project the best chance of becoming a genuinely strong trading system instead of an impressive demo.

---

# 40. Official Technology Notes Used for This Roadmap

- Laya currently supports domain fine-tuning and provides a workflow covering dataset construction, training, calibration, evaluation, and model publishing.
- Alpaca currently provides real-time WebSocket market data, stock snapshots, order APIs, and paper trading.
- Paper trading should be treated as execution simulation rather than proof of live profitability because live market impact, queue position, slippage, and other factors can differ.

---

**Project codename suggestion:** `ARGUS`

> Adaptive Real-time Global/US-market Intelligence System

The name is optional; the architecture and validation discipline are not.
