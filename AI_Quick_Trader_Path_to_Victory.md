# AI Quick Trader — Consolidated Path to Victory

**Supersedes as the execution roadmap:** neither source plan is deleted or replaced. This document consolidates and orders the work in `AI_Quick_Trader_Master_Plan.md` and `AI_Quick_Trader_Ultimate_Golden_Plan.md` into one authoritative program of work.

**Core objective:** build the strongest feasible short-horizon quantitative trading system we can, prove that every important component adds out-of-sample economic value after realistic costs, and allow progressively greater authority only when explicit promotion gates pass.

> Victory is not “the bot is complicated,” “the GUI looks institutional,” or “one backtest made money.” Victory is a system with repeatable positive net expectancy, controlled drawdown, realistic execution, safe abstention, operational reliability, auditability, and a promotion process that prevents unproven models from reaching meaningful capital.

There is no promise that the research will discover a durable profitable edge. If the evidence says the edge is absent, the correct outcome is to stop promotion, change the hypothesis, and re-research—not to force deployment.

---

# 1. The Two Plans, Unified

The **Master Plan** defines the minimum discipline required to become a real trading system:

- correct labels,
- serious historical data,
- causal features,
- calibrated multiclass alpha,
- expected-value ranking,
- regime awareness,
- Laya as a constrained meta-policy,
- portfolio allocation,
- realistic execution/backtesting,
- walk-forward and stress validation,
- paper/shadow/live progression,
- production risk,
- audit/replay,
- drift/model governance,
- dashboard and deployment.

The **Golden Plan** extends that foundation into a quantitative trading operating system:

- multi-horizon intelligence,
- specialist alpha models,
- strategy ensemble/meta-model,
- uncertainty and OOD detection,
- alpha-decay estimation,
- microstructure and order-flow features,
- execution optimization,
- advanced position/exit management,
- dynamic universe,
- news/events,
- champion/challenger governance,
- model registry and controlled retraining,
- professional GUI,
- service architecture, latency budgets, security, and production hardening.

The unified principle is:

```text
DATA TRUTH
   ↓
CAUSAL FEATURES
   ↓
CALIBRATED ALPHA
   ↓
ECONOMIC EV
   ↓
CONTEXT / REGIME / UNCERTAINTY
   ↓
SELECTIVE DECISION
   ↓
PORTFOLIO RISK
   ↓
REALISTIC EXECUTION
   ↓
CONTINUOUS POSITION MANAGEMENT
   ↓
AUDIT + REPLAY + LEARNING
```

Nothing gets promoted because it sounds intelligent. It gets promoted because it improves held-out economics and survives operational testing.

---

# 2. Definition of Victory

Joe Bot has reached the intended destination only when all five dimensions are strong at the same time.

## 2.1 Statistical victory

- Positive held-out expectancy over many chronological windows.
- No dependence on random train/test shuffling.
- Calibrated probabilities.
- No single-symbol or single-month dependency.
- Performance stable enough across regimes to be explainable.
- Model disagreement and OOD behavior measured rather than ignored.

## 2.2 Economic victory

- Positive net expectancy after spread, slippage, fees, delay, missed fills, and realistic order assumptions.
- Portfolio-level performance better than naive independent-trade deployment.
- Drawdown within predefined risk tolerance.
- Edge survives adverse-cost stress.
- Capital may remain idle when expected value is insufficient.

## 2.3 Operational victory

- No duplicate-order pathways.
- Broker/local order and position states reconcile.
- Disconnects, stale data, bad ticks, partial fills, and restarts have deterministic safe responses.
- Entry kill switches never block risk-reducing exits.
- Every decision is replayable.

## 2.4 Governance victory

- Every runtime artifact has a lineage: dataset, features, labels, model, calibration, EV state, validation metrics, and code commit.
- Champion/challenger promotion is controlled.
- Retraining cannot silently replace a production model.
- Failed components are removed rather than protected for prestige.

## 2.5 Product victory

- Operator can understand what Joe sees, why it acts, how much risk it carries, whether models are healthy, and what changed.
- GUI communicates risk and uncertainty, not excitement.
- Research, production, replay, execution, and model state are connected through a coherent platform.

---

# 3. Current State — October 2026

The codebase has advanced rapidly. Engineering exists through the shadow/live-candidate boundary, but empirical promotion is far behind the implementation frontier.

## 3.1 Engineering already built

- Immutable/versioned Parquet research data layer and manifests.
- Triple-barrier multiclass labeling.
- 58-feature causal Feature Engine V2.
- SPY/QQQ market context, breadth, relative strength, time-normalized activity.
- Multiclass model training and model-family benchmark harness.
- Probability calibration and classification diagnostics.
- Event-driven V0.6 trade backtester.
- Spread, slippage, delay, fees, tick and execution stress assumptions.
- Chronological walk-forward framework.
- Structural + empirical V0.7 expected-value model.
- EV ranking, risk-based sizing, correlation-aware portfolio allocation.
- Laya historical dataset generator, runtime adapter, calibration, constrained veto gate, and gated-vs-ungated walk-forward harness.
- Paper-only broker abstraction and Alpaca paper adapter.
- Order state management, position management, reconciliation, exit engine, kill switches.
- Append-only SQLite audit/replay.
- V0.95 local-only shadow broker and shadow runtime.
- Live Feature Engine V2 state seeded from history.
- Frozen alpha runtime loader.
- Live V0.5 → V0.7 → optional V0.8 candidate-provider seam.
- 89 offline tests currently passing.

## 3.2 Engineering only partially built

- Exit intelligence is deterministic target/stop/time; no learned edge-decay exit model yet.
- Correlation controls exist, but richer factor/beta clustering can improve.
- Stress testing covers execution-cost scenarios, but not the full failure campaign.
- Metrics are useful but incomplete for MFE/MAE, CVaR, regime attribution, and execution diagnostics.
- Live scanner exists conceptually through provider/universe plumbing, but not the mature 300–1000 symbol staged scanner.
- Paper/shadow infrastructure exists, but proof campaigns have not been run.

## 3.3 Major capabilities not yet built

- Serious market-regime engine.
- Multi-horizon prediction stack.
- Specialist strategy models: momentum, breakout, pullback, mean reversion, order flow, event model.
- Ensemble/meta-model selecting specialist weights.
- Uncertainty engine and out-of-distribution detection.
- Alpha-decay estimator.
- Full quote/microstructure/order-flow feature layer.
- Execution policy optimizer: passive limit / marketable limit / market / cancel-reprice.
- Learned exit/position-management intelligence.
- Model registry and formal artifact-promotion service.
- Controlled retraining pipeline.
- Champion/challenger production framework.
- Dynamic 300–1000+ symbol universe.
- Structured event/news intelligence.
- Professional command-center GUI.
- Full REST/WebSocket backend contract.
- Production service split, redundancy, disaster recovery, and hardened security.

## 3.4 Most important unresolved fact

We do **not yet have real evidence that the complete system possesses durable positive net expectancy**.

Synthetic/unit-test fixtures demonstrate correctness of code pathways, not market edge. Therefore the next decisive phase is proof, not feature accumulation.

---

# 4. The Consolidated Architecture

The final system should evolve toward:

```text
                         MARKET / EVENT INPUTS
             ┌───────────────┼────────────────┐
             │               │                │
       historical bars   live trades/quotes   events/news
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    DATA QUALITY / INGEST
                             │
                             ▼
                      DYNAMIC UNIVERSE
                             │
                             ▼
                    CHEAP ACTIVITY SCOUT
                             │
                             ▼
                     LIVE FEATURE STORE
                             │
          ┌──────────────────┼───────────────────┐
          ▼                  ▼                   ▼
       REGIME           MICROSTRUCTURE      MARKET CONTEXT
          │                  │                   │
          └──────────────────┼───────────────────┘
                             ▼
                 MULTI-HORIZON SPECIALISTS
             ┌───────┬───────┬───────┬────────┐
             │       │       │       │        │
          momentum breakout pullback reversion order-flow
             └───────────────┬────────────────┘
                             ▼
                       META-MODEL
                             │
                  calibrated probabilities
                             │
                             ▼
                         EV ENGINE
                             │
                    uncertainty / OOD
                             │
                             ▼
                    TOP CANDIDATES ONLY
                             │
                             ▼
                  OPTIONAL LAYA VETO/POLICY
                             │
                             ▼
                     PORTFOLIO OPTIMIZER
                             │
                             ▼
                    HARD PRODUCTION RISK
                             │
                             ▼
                    EXECUTION OPTIMIZER
                             │
                             ▼
                           BROKER
                             │
                             ▼
                      OPEN POSITIONS
                             │
                             ▼
                POSITION / EXIT INTELLIGENCE
                             │
                             ▼
                  AUDIT → REPLAY → RESEARCH
                             │
                             ▼
             MODEL REGISTRY / CHAMPION-CHALLENGER
```

The GUI sits across all layers as an operator surface, not as part of the decision authority.

---

# 5. One Roadmap — Ten Victory Stages

The Master phases and GOLD phases are consolidated below into a dependency-safe order. Previously built later-stage infrastructure is preserved; it simply remains **unpromoted** until earlier gates pass.

---

# VICTORY STAGE 0 — Freeze, Inventory, Reproducibility

**Source-plan coverage:** Master V0.3 freeze / Phase B manifests / Golden GOLD 0 / Model Registry foundations.

## Goal

Make every experiment and runtime artifact reproducible before expensive research begins.

## Required work

- Create a formal experiment/run manifest.
- Record code commit, dataset version/hash, label config, feature version, model config, calibration config, EV config, execution assumptions, allocator limits, and random seeds.
- Add an artifact factory that packages everything required by live/shadow inference.
- Create model-registry schema with statuses:
  - EXPERIMENTAL
  - CHALLENGER
  - CHAMPION
  - RETIRED
- Add one promotion command that refuses incomplete artifacts.
- Remove repository hygiene hazards and ensure no secrets are tracked.

## Deliverable

```text
artifacts/<candidate_id>/
  runtime_manifest.json
  alpha_model.pkl
  feature_schema.json
  dataset_manifest.json
  ev_state.*
  correlations.csv
  risk_config.json
  execution_config.json
  validation_report.json
  optional_laya/
```

## Gate

A clean checkout can reproduce the candidate's evaluation and load the exact same runtime package.

## Current status

**PARTIAL.** Dataset/model manifests exist, but no unified promotion package/registry yet.

---

# VICTORY STAGE 1 — Prove the Core Alpha Exists

**Source-plan coverage:** Master A–E, N–Q, O/P; GOLD 1–7; Golden validation/backtesting sections.

This is the immediate critical path.

## 1A. Build/freeze the production research dataset

Target initially:

- 2–5 years of minute data,
- at least the current 50-symbol liquid universe,
- SPY and QQQ context,
- explicit corporate-action adjustment mode,
- quote data later when available,
- immutable raw version.

Do not expand to 300–1000 symbols until the 50-symbol core pipeline is economically understood.

## 1B. Label research

Benchmark multiple label regimes:

- fixed triple barriers,
- ATR/volatility-aware barriers,
- several horizons,
- ambiguity/no-resolution statistics.

No label choice is accepted because its class balance looks attractive.

## 1C. Feature ablation

The current 58-feature V2 schema becomes a hypothesis, not sacred truth.

Run ablations for:

- market context,
- breadth,
- relative strength,
- time-normalized volume,
- trend blocks,
- volatility blocks,
- session/time blocks.

Remove feature groups that add no stable out-of-sample value.

## 1D. Model tournament

Benchmark at minimum:

- XGBoost,
- HistGradientBoosting,
- LightGBM when available,
- CatBoost when available.

Measure:

- LONG/SHORT precision/recall,
- macro F1,
- PR-AUC,
- log loss,
- multiclass Brier,
- calibration error,
- economic EV by confidence bucket.

## 1E. Calibration

Calibrate using chronological calibration partitions only.

Confidence must correspond sensibly to realized outcome frequency. Uncalibrated confidence cannot drive sizing.

## 1F. Event-driven economic validation

Run V0.6 under:

- realistic spread,
- realistic slippage,
- one-or-more-bar entry delay scenarios,
- fees,
- conservative same-bar target/stop assumptions,
- close/session handling,
- stress scenarios.

## 1G. EV validation

Prove:

```text
higher predicted net EV
        ↓
monotonically better realized net return
```

If EV ranking does not separate realized economics, V0.7 cannot be promoted.

## 1H. Walk-forward campaign

Run enough rolling windows to cover multiple market conditions.

No final threshold is repeatedly tuned against the final held-out set.

## Promotion gate: CORE EDGE

Must demonstrate, on real data:

- positive net out-of-sample expectancy under conservative costs,
- stability across multiple walk-forward windows,
- useful calibration,
- no single-symbol dependency,
- acceptable drawdown,
- edge remains positive under cost stress,
- higher predicted EV corresponds to better realized results.

If this gate fails, **stop here and research the alpha**. Do not hide the failure behind Laya, GUI work, or execution sophistication.

## Current status

**ENGINEERING BUILT; EMPIRICAL GATE NOT RUN.**

---

# VICTORY STAGE 2 — Context, Regime, Multi-Horizon and Specialist Intelligence

**Source-plan coverage:** Master F, I, Z; GOLD 8–9; Golden multi-horizon/specialist/meta-model/cross-sectional/breadth sections.

## 2A. Regime engine

Classify at least:

```text
TREND_UP
TREND_DOWN
RANGE
HIGH_VOL
LOW_VOL
SHOCK
OPENING_VOLATILITY
```

Use market momentum, realized volatility, breadth, dispersion, range, and volume.

Regime output must be probabilistic/confidence-aware rather than a brittle hard label where possible.

## 2B. Multi-horizon intelligence

Add horizon-specific outputs where data supports them:

```text
30 sec / 1 min / 2 min / 5 min / 10 min / 20 min
```

Use multi-horizon structure to estimate:

- entry urgency,
- alpha decay,
- expected hold duration,
- target aggressiveness,
- execution style.

## 2C. Specialist strategies

Build separate research interfaces for:

- momentum continuation,
- breakout/opening range,
- pullback continuation,
- mean reversion,
- later: order flow,
- later: event reaction.

Each specialist emits the same contract:

```text
P(WAIT), P(LONG), P(SHORT), EV, uncertainty
```

## 2D. Meta-model

Inputs:

- specialist probabilities,
- regime,
- liquidity/volatility,
- breadth/context,
- recent specialist performance,
- event state later.

Outputs strategy weights + ensemble probabilities + ensemble EV.

## Promotion gate: INTELLIGENCE EXPANSION

The regime-conditioned multi-strategy system must beat the best standalone core model out-of-sample **after costs**, or it does not replace it.

## Current status

**MOSTLY NOT STARTED.** Existing `breakout.py`/`pullback.py` do not constitute the Golden specialist ensemble.

---

# VICTORY STAGE 3 — Laya, Uncertainty and OOD

**Source-plan coverage:** Master G; GOLD 10–11 and 25; Golden uncertainty/OOD sections.

## 3A. Finish the real Laya campaign

The infrastructure already exists. Now use the real historical states generated from promoted quant candidates.

Train/validate/test chronologically.

Laya initially remains asymmetric:

```text
quant rejects → Laya cannot create trade
quant approves + Laya WAIT/risk veto → reject
```

Add `execution_urgency` only after the base decision heads are proven.

## 3B. Model disagreement

Track disagreement across specialist/ensemble models.

High disagreement should lower size or force WAIT.

## 3C. OOD engine

Benchmark approaches such as:

- feature percentile violations,
- Mahalanobis-style distance,
- Isolation Forest,
- density/embedding distance,
- ensemble disagreement.

Severe OOD becomes a hard no-entry condition.

## Promotion gate: META SAFETY

- Laya must improve held-out EV/drawdown/trade quality versus identical no-Laya baseline.
- OOD/uncertainty gating must improve tail behavior or reduce failure exposure without destroying the economic edge.
- Otherwise reduce/remove those components.

## Current status

**Laya engineering built; real fine-tuning/validation pending. Uncertainty/OOD not built.**

---

# VICTORY STAGE 4 — Portfolio, Position and Exit Excellence

**Source-plan coverage:** Master J–L/K; GOLD 12–13; Golden position manager, optimizer, correlation clusters.

## 4A. Upgrade portfolio optimizer

Current V0.7 is a strong foundation. Extend only when justified:

- factor/beta exposure,
- richer correlation clusters,
- regime-dependent limits,
- liquidity-aware caps,
- uncertainty-aware sizing,
- severe cap on any future fractional-Kelly experiment.

## 4B. Position state

For every position continuously maintain:

- entry/size/current price,
- unrealized PnL,
- MFE/MAE,
- age,
- original/current EV,
- original/current model state,
- regime change,
- spread/liquidity deterioration,
- stop/target.

## 4C. Exit intelligence

The existing deterministic exit engine remains the hard safety baseline.

Research a separate model for:

```text
HOLD
REDUCE_25
REDUCE_50
EXIT
TAKE_PROFIT
```

Hard stops are never overridden.

## 4D. Alpha decay

Estimate how quickly edge decays after the signal. Feed it into exit and execution policy.

## Promotion gate: POSITION ECONOMICS

Advanced portfolio/exit logic must improve realized risk-adjusted outcomes versus fixed target/stop + current allocator on held-out periods.

## Current status

**Portfolio base built. Advanced position/exit intelligence not built.**

---

# VICTORY STAGE 5 — Microstructure and Execution Intelligence

**Source-plan coverage:** Master C7/M/N/P; GOLD 14–15; Golden execution intelligence/reliability.

## 5A. Quote/trade data layer

Add and version:

- bid/ask,
- spread,
- depth where available,
- imbalance,
- trade flow,
- quote/trade activity,
- short-horizon liquidity state.

## 5B. Microstructure features

Only retain features with measurable short-horizon value after costs.

## 5C. Fill/partial-fill realism

Extend backtester for:

- partial fills,
- queue/fill probability approximations where defensible,
- missed orders,
- stale/cancel/reprice behavior,
- halts and exchange/broker edge cases.

## 5D. Execution optimizer

Compare:

```text
PASSIVE_LIMIT
MARKETABLE_LIMIT
MARKET
CANCEL_REPRICE
WAIT
```

Inputs include spread, liquidity, alpha decay, urgency, volatility, and size.

## Promotion gate: EXECUTION EDGE PRESERVATION

Execution optimizer must reduce realized cost without unacceptable missed-trade rate and without introducing state risk.

## Current status

**Streaming/order state foundation built; advanced microstructure/execution optimization not built.**

---

# VICTORY STAGE 6 — Artifact Factory, Model Registry and Controlled Promotion

**Source-plan coverage:** Master X/Y/deployment; Golden 30–32.

This stage converts research into a governed production candidate.

## 6A. Artifact factory

One command produces a candidate package only after required reports exist.

## 6B. Registry

Store:

```text
model_id
model_type
dataset_version
feature_version
label_version
training period
validation metrics
walk-forward metrics
stress metrics
paper metrics
shadow metrics
live metrics
status
```

## 6C. Promotion policy

```text
EXPERIMENTAL
    ↓ evidence
CHALLENGER
    ↓ shadow/paper superiority
CHAMPION
    ↓ degradation/replacement
RETIRED
```

No automatic promotion after retraining.

## 6D. Drift detection

Monitor:

- feature drift,
- prediction distribution drift,
- calibration drift,
- EV drift,
- execution-cost drift,
- regime frequency changes.

## Promotion gate: GOVERNANCE

Production runtime can only load a complete, validated, registry-approved artifact package.

## Current status

**NOT YET BUILT AS A COMPLETE SYSTEM.**

---

# VICTORY STAGE 7 — Prove the System in Real Time Without Capital

**Source-plan coverage:** Master R/S; GOLD 18–20; Golden reliability/state reconciliation.

We already built both V0.9 paper infrastructure and V0.95 shadow infrastructure. Validation now becomes two proof tracks that may run in parallel after Stage 1 core-edge promotion.

## 7A. Shadow intelligence proof

Live market data + full promoted intelligence + local-only simulated execution.

Measure:

- predicted vs realized EV,
- expected vs observed fill/slippage behavior,
- latency,
- feature parity,
- regime distribution,
- OOD frequency,
- model failures,
- stale/late bars,
- decision reproducibility.

Zero external broker orders.

## 7B. Paper execution proof

Alpaca paper execution with promoted candidate stack.

Target:

- hundreds of trades,
- multiple sessions/regimes,
- reconnect/restart tests,
- no unexplained account/order/position mismatch,
- no duplicate orders,
- partial-fill behavior verified,
- kill switches exercised deliberately.

## 7C. Failure campaign

Inject/test:

- stale feed,
- WebSocket reconnect,
- API timeout,
- duplicate/out-of-order message,
- bad tick,
- broker rejection,
- partial fill,
- local/broker position mismatch,
- runtime restart,
- model unavailable,
- NaN feature.

## Promotion gate: REAL-TIME PROOF

Both decision assumptions and execution/state behavior remain compatible with research expectations across a meaningful sample.

## Current status

**INFRASTRUCTURE BUILT; REAL PROOF NOT RUN.**

---

# VICTORY STAGE 8 — Operator OS / Golden GUI

**Source-plan coverage:** Master W/V; GOLD 16–17; Golden GUI 35–50, backend/frontend contract.

The GUI can be developed once the underlying schemas are stable, but it does not unblock capital by itself.

## GUI V1

- Command Center
- Live Scanner
- Portfolio
- Positions
- Risk Center
- System Health

## GUI V2

- Symbol detail drawer
- Model Console
- Laya Console
- Research Lab
- Backtest Explorer
- Execution Console
- Decision Replay
- Model registry/champion-challenger views

## Backend contract

REST for configuration/research/history.

WebSocket for:

- scanner,
- positions,
- orders,
- risk,
- system/model health.

## Visual principle

Professional, restrained, risk-first. No casino UI.

## Current status

**NOT BUILT.**

---

# VICTORY STAGE 9 — Expansion: Dynamic Universe, Events, Champion/Challenger

**Source-plan coverage:** Master H/I/X/Y/Z; GOLD 22–25.

This is expansion after the core system has evidence.

## 9A. Champion/challenger

Run candidate models beside champion without silent promotion.

## 9B. Dynamic universe

Expand from 50 core symbols toward 300–1000+ liquid stocks through a staged scanner:

```text
large universe
→ cheap liquidity/activity filter
→ feature/alpha
→ EV
→ uncertainty/Laya
→ portfolio
```

## 9C. News/events

Convert earnings, guidance, filings, analyst actions, macro and other events into structured features.

LLM/NLP output informs models; it never directly submits orders.

## 9D. Cross-sectional intelligence

Improve breadth, dispersion, sector/factor and relative-strength state.

## Promotion gate

Every expansion must improve held-out net economics or operational resilience. More symbols/models are not automatically better.

## Current status

**NOT BUILT.**

---

# VICTORY STAGE 10 — Production Hardening, Tiny Live, Controlled Scale

**Source-plan coverage:** Master T/U/deployment/security; GOLD 21/26 and scale rule.

This is deliberately last.

## 10A. Production hardening

- redundant market-data strategy,
- watchdog/supervisor,
- persistent state recovery,
- encrypted secrets,
- GUI authentication/RBAC,
- database backups,
- disaster recovery,
- structured alerts,
- deployment automation,
- explicit latency telemetry,
- service health/SLOs.

Do not split into microservices until profiling/operational needs justify it. A modular monolith is preferable early.

## 10B. Tiny live

Only after all mandatory go-live gates.

Use capital whose total loss is acceptable.

The purpose is to validate real execution assumptions, not to maximize profit.

## 10C. Controlled scale

Capital increases only in predefined steps after sufficient live evidence.

Never scale because of a short winning streak.

## Gate

Realized execution, slippage, state reliability, drawdown, and expectancy stay inside predefined expected ranges.

## Current status

**NOT STARTED — CORRECTLY.**

---

# 6. Critical Path vs Parallel Work

Not every capability should block every other task. We therefore separate the program into a critical path and safe parallel work.

## Critical path to any live capital

```text
Stage 0 reproducibility
        ↓
Stage 1 real core-edge proof
        ↓
Stage 2 regime/specialist research (only promoted if additive)
        ↓
Stage 3 Laya/uncertainty proof (only promoted if additive)
        ↓
Stage 4 portfolio/exit proof
        ↓
Stage 5 execution realism
        ↓
Stage 6 governed artifact promotion
        ↓
Stage 7 shadow + paper proof
        ↓
Stage 10 tiny live
```

## Parallel but non-blocking

After Stage 1 schemas stabilize:

- GUI work,
- replay UX,
- model registry UI,
- dynamic-universe research,
- news/event research,
- production service packaging.

None of these may be used to bypass the evidence gates.

---

# 7. Exact Next Build Order From Today

This is the actionable sequence from the current repository state.

## NOW — Sprint 1: Artifact Factory + Data Reality

1. Build candidate artifact manifest/registry skeleton.
2. Freeze one real immutable 2+ year / 50-symbol dataset.
3. Generate dataset quality report and hash.
4. Run barrier/label sweep including volatility-aware candidates.
5. Select label configurations for the research tournament without touching final test periods.

## Sprint 2: Core Alpha Tournament

6. Run feature ablations.7. Benchmark XGBoost / HistGradientBoosting / LightGBM / CatBoost where available.
8. Calibrate finalists.
9. Run chronological walk-forward classification diagnostics.
10. Freeze the best challenger candidate(s).

## Sprint 3: Economic Proof

11. Run V0.6 realistic execution walk-forward.
12. Stress spread/slippage/delay/fees.
13. Measure EV monotonicity by predicted EV bucket.
14. Fit V0.7 empirical EV only on earlier periods.
15. Run portfolio walk-forward versus V0.6 baseline.
16. Produce the first **Core Edge Report** with explicit PASS/FAIL.

**If Sprint 3 fails: return to labels/features/models. Do not continue down the promotion path.**

## Sprint 4: Regime + Specialist Intelligence

17. Build regime engine.
18. Measure core model performance by regime.
19. Implement proper specialist interfaces.
20. Build momentum/breakout/pullback/reversion specialists.
21. Build ensemble/meta-model.
22. Promote only if held-out economics improve.

## Sprint 5: Laya + Uncertainty

23. Build real Laya dataset from promoted quant stack.
24. Fine-tune and calibrate Laya.
25. Run gated vs ungated held-out economics.
26. Build ensemble disagreement + OOD baseline.
27. Promote only additive components.

## Sprint 6: Position/Exit Intelligence

28. Add MFE/MAE/live position telemetry.
29. Build alpha-decay estimates.
30. Train/evaluate exit model against deterministic baseline.
31. Extend portfolio risk with factor/beta/uncertainty if evidence warrants.

## Sprint 7: Microstructure + Execution

32. Acquire/version quote/trade data.
33. Build microstructure features.
34. Extend partial-fill/missed-fill simulator.
35. Benchmark execution policies.
36. Promote execution optimizer only if realized-cost improvement is clear.

## Sprint 8: Promotion Factory

37. Complete model registry.
38. Build artifact factory packaging alpha/EV/correlation/risk/execution/Laya.
39. Add promotion gate CLI.
40. Add drift baselines to promoted artifacts.

## Sprint 9: Shadow + Paper Campaign

41. Run multi-session full-intelligence shadow.
42. Analyze expected-vs-realized drift.
43. Run full paper trading.
44. Execute fault-injection campaign.
45. Require hundreds of clean paper trades and reconciled state.

## Sprint 10: Golden GUI

46. Build backend REST/WebSocket API.
47. Build Command Center/Scanner/Portfolio/Risk/System Health.
48. Add Research/Backtest/Replay/Model/Laya/Execution consoles.

The GUI can begin earlier once contracts stabilize, but it is listed here to keep the critical engineering sequence unambiguous.

## Sprint 11: Expansion

49. Champion/challenger automation.
50. Dynamic universe to 300–1000+.
51. News/event features.
52. Advanced uncertainty/OOD.

## Sprint 12: Production Hardening + Tiny Live

53. Redundancy/security/recovery/deployment hardening.
54. Formal go-live review.
55. Tiny-live experiment.
56. Compare realized execution against paper/shadow assumptions.
57. Scale only through predefined evidence-based steps.

---

# 8. Mandatory Promotion Gates

These gates resolve all ambiguity between the Master and Golden plans.

## Gate A — Data

Pass only if:

- immutable dataset build is reproducible,
- quality issues are quantified,
- no forward-filled fake minute bars,
- timestamp/session handling is correct,
- corporate-action assumptions are explicit.

## Gate B — Alpha

Pass only if:

- chronological held-out separation exists,
- probabilities are useful/calibrated,
- results are not one-symbol dependent,
- feature/model choice survives multiple windows.

## Gate C — Economics

Pass only if:

- net expectancy is positive after conservative costs,
- EV ranking has realized monotonicity,
- drawdown is acceptable,
- adverse execution stress does not erase the edge.

## Gate D — Added Intelligence

Regime, specialists, ensemble, Laya, uncertainty, OOD, and advanced exits are independently compared against an identical baseline. A component that does not improve economics or safety is not promoted.

## Gate E — Operational

Pass only if:

- duplicate orders are impossible/idempotent,
- disconnect/reconnect succeeds,
- partial fills/rejections are handled,
- local/broker state reconciles,
- kill switches are tested,
- exits remain possible during entry halts.

## Gate F — Shadow/Paper

Pass only after a meaningful multi-session sample demonstrates that live features, latency, execution assumptions, and state behavior are consistent with research expectations.

## Gate G — Tiny Live

Pass only if real fills/slippage/state remain within predefined ranges and losses/drawdowns remain within risk controls.

---

# 9. Scorecard — Master + Golden Consolidated

Legend:

- **BUILT** = engineering exists and tests pass.
- **PARTIAL** = useful foundation exists but plan target is incomplete.
- **UNVALIDATED** = engineering exists but required real-market evidence is missing.
- **NOT STARTED** = meaningful implementation absent.

| Capability | Status | Immediate requirement |
|---|---|---|
| Triple-barrier multiclass labels | BUILT / UNVALIDATED | real label study + ATR-aware comparison |
| Historical Parquet lake + manifests | BUILT / UNVALIDATED | freeze real multi-year 50-symbol production dataset |
| Feature Engine V2 | BUILT / UNVALIDATED | ablations on real dataset |
| SPY/QQQ/breadth/relative strength | BUILT / UNVALIDATED | prove additive EV |
| Probability calibration | BUILT / UNVALIDATED | real chronological calibration campaign |
| Model-family benchmarks | BUILT / UNVALIDATED | run full tournament |
| Realistic event backtester | BUILT / UNVALIDATED | real-data economic proof |
| Walk-forward | BUILT / UNVALIDATED | multi-window campaign |
| Execution stress | PARTIAL / UNVALIDATED | expand failure + partial-fill scenarios |
| EV model | BUILT / UNVALIDATED | prove EV monotonicity |
| Portfolio allocator | BUILT / UNVALIDATED | real portfolio walk-forward |
| Correlation exposure | PARTIAL / UNVALIDATED | factor/beta + real validation |
| Laya dataset/gate/calibration | BUILT / UNVALIDATED | fine-tune real specialist + A/B proof |
| Market regime engine | NOT STARTED | build before mature strategy ensemble |
| Multi-horizon intelligence | NOT STARTED | research after core alpha proof |
| Strategy specialists | NOT STARTED | momentum/breakout/pullback/reversion |
| Meta-model/ensemble | NOT STARTED | specialist weighting |
| Uncertainty engine | NOT STARTED | disagreement/calibration confidence |
| OOD detection | NOT STARTED | fail-safe distribution-shift detector |
| Alpha decay | NOT STARTED | time-decay estimates |
| Learned exit model | NOT STARTED | compare with deterministic exit baseline |
| Deterministic exit safety | BUILT | retain regardless of ML exit model |
| Quote/microstructure layer | NOT STARTED | acquire/version quote/trade data |
| Execution optimizer | NOT STARTED | benchmark policy choices |
| WebSocket market state | BUILT | multi-hour proof |
| Paper broker/order lifecycle | BUILT / UNVALIDATED | hundreds of real paper trades |
| Position reconciliation | BUILT / UNVALIDATED | live fault campaign |
| Production risk kill switches | BUILT / UNVALIDATED | deliberate trigger/recovery tests |
| Audit/replay | BUILT | enrich model/version/raw-state linkage |
| Shadow broker/runtime | BUILT / UNVALIDATED | multi-session full-stack run |
| Live 58-feature store | BUILT / UNVALIDATED | live/research parity monitoring |
| Live V0.5→V0.7→V0.8 provider | BUILT / UNVALIDATED | promoted artifacts + shadow proof |
| Artifact factory | NOT STARTED | highest-priority build |
| Model registry | NOT STARTED | highest-priority governance build |
| Drift detection | NOT STARTED | add before long-running production |
| Champion/challenger | NOT STARTED | post-core proof |
| Dynamic universe | NOT STARTED | post-core proof |
| News/events | NOT STARTED | later expansion |
| Golden GUI | NOT STARTED | build once schemas/contracts stabilize |
| Production hardening | PARTIAL | redundancy/security/recovery later |
| Tiny live | NOT STARTED | blocked by all mandatory gates |
| Controlled scaling | NOT STARTED | blocked by live evidence |

---

# 10. Victory Metrics Dashboard

No single metric determines promotion. Every candidate should report at minimum:

## Model quality

- macro F1,
- LONG/SHORT PR-AUC,
- log loss,
- Brier score,
- ECE/calibration plots,
- confidence-bucket realized rates.

## Economic quality

- net expectancy bps/trade,
- expected-vs-realized EV calibration,
- profit factor,
- total return,
- risk-adjusted return,
- positive-window fraction,
- LONG/SHORT decomposition,
- symbol/sector/regime decomposition.

## Risk

- max drawdown,
- drawdown duration,
- tail-loss quantiles,
- CVaR where appropriate,
- exposure concentration,
- correlation-cluster exposure,
- worst regime/window.

## Trade quality

- win rate only as context,
- MFE,
- MAE,
- holding time,
- target/stop/time/exit-model attribution,
- edge decay by time since signal.

## Execution

- spread paid,
- slippage,
- signal-to-order latency,
- signal-to-fill latency,
- fill rate,
- partial-fill rate,
- missed-trade rate,
- rejection/cancel rate.

## Operational

- runtime crashes,
- reconnects,
- stale-feed incidents,
- mismatches,
- duplicate-order attempts,
- kill-switch activations,
- replay completeness.

---

# 11. Non-Negotiable Safety and Research Rules

1. Never optimize only for win rate.
2. Never randomly shuffle time-series market rows for final evaluation.
3. Never allow Laya/LLMs to directly submit arbitrary orders.
4. Never use uncalibrated confidence as risk sizing probability.
5. Never force capital deployment.
6. Never assume paper fills equal live fills.
7. Never retune against the final held-out test set repeatedly.
8. Never let hard stops be overridden by AI conviction.
9. Never automatically average down outside predefined strategy rules.
10. Never silently promote a retrained model.
11. Never allow missing features/models/EV state to degrade into “best effort” trading; fail closed.
12. Never add complexity unless it improves economics, safety, or operational resilience.
13. Never increase capital because of a short winning streak.
14. Never declare profitability from unit tests or synthetic fixtures.
15. Never treat a sophisticated architecture as evidence of edge.

---

# 12. The Path to Victory, in One View

```text
TODAY
│
├── We have a powerful engineering skeleton through live shadow intelligence.
│
▼
FREEZE REPRODUCIBLE ARTIFACTS
│
▼
BUILD THE REAL DATASET
│
▼
PROVE CORE ALPHA + CALIBRATION + EV
│        └── FAIL → research labels/features/models again
│
▼
ADD REGIME + SPECIALISTS + ENSEMBLE
│        └── only keep additive components
│
▼
FINE-TUNE LAYA + UNCERTAINTY/OOD
│        └── only keep additive/safety-positive components
│
▼
ADVANCED PORTFOLIO + EXIT / ALPHA DECAY
│
▼
MICROSTRUCTURE + EXECUTION OPTIMIZATION
│
▼
PACKAGE A GOVERNED CHAMPION ARTIFACT
│
├── SHADOW PROOF ─────┐
│                     ├── both clean
└── PAPER PROOF ──────┘
          │
          ▼
    GOLDEN GUI / OPERATOR OS
          │
          ▼
  PRODUCTION HARDENING
          │
          ▼
      TINY LIVE
          │
          ▼
   EVIDENCE REVIEW
          │
          ▼
CHAMPION/CHALLENGER + DYNAMIC UNIVERSE + EVENTS
          │
          ▼
CONTROLLED SCALE ONLY WHEN EVIDENCE SUPPORTS IT
```

---

# 13. Immediate Mission

The next milestone is **not V1.0**.

It is:

> **VICTORY MILESTONE 1 — Produce Joe Bot's first reproducible Core Edge Report from a real multi-year dataset.**

That report is the point where we stop asking “how smart is the architecture?” and start answering:

```text
Does Joe actually have edge?
Where does it have edge?
When does it lose edge?
How much survives realistic costs?
How stable is it across time/regimes/symbols?
Can predicted EV be trusted?
```

Once that is answered convincingly, every Golden upgrade has a baseline to beat.

That is the path to victory.
