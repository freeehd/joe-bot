# Joe Bot — AI Quick Trader

Joe Bot is an experimental short-horizon US-equities trading research project.
The repository is currently being upgraded from the v0.3 dual-binary prototype
toward the V0.4 **Correct Intelligence Foundation** in
`AI_Quick_Trader_Master_Plan.md`.

## Safety state

- Research / paper analysis only.
- No automatic live trading is enabled.
- The new V0.4 pipeline does not submit orders.

## Current research architecture

```text
minute bars
   ↓
session-safe v0.3 features
   ↓
triple-barrier labels
   ↓
WAIT / LONG / SHORT multiclass XGBoost
   ↓
chronological calibration
   ↓
out-of-sample confidence analysis
```

The legacy scanner remains available while the V0.4 model is researched and
validated. It should not be treated as evidence of profitability.

## V0.4 training

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env` and add **paper/data** Alpaca credentials.
4. Run:

```bash
python -m models.train_multiclass
```

The V0.4 trainer:

- downloads a multi-symbol minute-bar dataset,
- resets features at session boundaries,
- creates target-before-stop triple-barrier labels,
- maps outcomes to `WAIT=0`, `LONG=1`, `SHORT=2`,
- purges the label horizon around chronological validation boundaries,
- trains a class-balanced multiclass XGBoost model,
- calibrates probabilities on a later holdout,
- evaluates only on the final untouched test period.

## Offline tests

```bash
python -m unittest discover -s tests -v
```

Broker scripts under `execution/` are legacy integration scripts and are not
part of the offline unit-test suite.
