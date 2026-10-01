"""Live V0.5 -> V0.7 -> optional V0.8 candidate provider for shadow production."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from live.feature_store import LiveFeatureStore
from live.paper_engine import EntryProposal, PaperEngineState
from market.sectors import sector_for
from market.stream import BarEvent
from models.live_alpha import LiveAlphaModel
from strategy.ev_model import EVConfig, EmpiricalEVModel
from strategy.laya_gate import LayaVetoGate
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2, PortfolioConstraints


@dataclass(frozen=True)
class LiveProviderConfig:
    minimum_direction_confidence: float = 0.0
    min_net_ev_bps: float = 0.0


class LiveCandidateProvider:
    def __init__(
        self,
        feature_store: LiveFeatureStore,
        alpha_model: LiveAlphaModel,
        ev_model: EmpiricalEVModel,
        *,
        allocator: PortfolioAllocatorV2 | None = None,
        correlations: pd.DataFrame | None = None,
        laya_gate: LayaVetoGate | None = None,
        config: LiveProviderConfig | None = None,
    ) -> None:
        self.features = feature_store
        self.alpha = alpha_model
        self.ev_model = ev_model
        self.ranker = EVRanker(ev_model)
        self.allocator = allocator or PortfolioAllocatorV2(trade_config=ev_model.trade_config)
        self.correlations = correlations
        self.laya_gate = laya_gate
        self.config = config or LiveProviderConfig()
        self.last_diagnostics: dict = {}

    @staticmethod
    def _current_positions(state: PaperEngineState) -> list[dict]:
        positions = []
        for item in state.positions:
            allocation = float(item.get("notional", 0.0))
            stop = item.get("stop_price")
            entry = float(item.get("average_entry_price", 0.0))
            qty = abs(float(item.get("quantity", 0.0)))
            risk = abs(entry - float(stop)) * qty if stop is not None else 0.0
            positions.append({
                "symbol": item["symbol"], "direction": item["direction"],
                "sector": item.get("sector", "UNKNOWN"), "allocation": allocation, "risk_dollars": risk,
            })
        return positions

    def __call__(self, event: BarEvent, state: PaperEngineState) -> Sequence[EntryProposal]:
        try:
            batch = self.features.add_bar(event)
            if batch is None:
                return []
            frame = batch.frame.copy()
            candidate_rows = frame.loc[~frame["training_symbol"].isin(["SPY", "QQQ"])].copy()
            predictions = self.alpha.predict_frame(candidate_rows)
            candidates: list[dict] = []
            for (_, row), prediction in zip(candidate_rows.iterrows(), predictions):
                if prediction["direction"] not in {"LONG", "SHORT"}:
                    continue
                if float(prediction["confidence"]) < self.config.minimum_direction_confidence:
                    continue
                symbol = str(row["training_symbol"])
                item = row.to_dict()
                item.update(prediction)
                item.update({"symbol": symbol, "price": float(row["close"]), "sector": sector_for(symbol)})
                candidates.append(item)

            current = self._current_positions(state)
            ranked = self.ranker.rank(
                candidates,
                correlations=self.correlations,
                current_positions=current,
                min_net_ev_bps=self.config.min_net_ev_bps,
            )
            gate_result = None
            if self.laya_gate is not None:
                gate_result = self.laya_gate.gate_candidates(
                    ranked,
                    timestamp=batch.timestamp,
                    account_equity=state.account.equity,
                    current_positions=current,
                    correlations=self.correlations,
                )
                ranked = gate_result["approved"]

            allocated = self.allocator.allocate(
                ranked,
                account_equity=state.account.equity,
                correlations=self.correlations,
                current_positions=current,
            )
            proposals: list[EntryProposal] = []
            for selected in allocated["positions"]:
                price = float(selected["price"])
                stop_pct = float(selected.get("stop_pct", self.ev_model.trade_config.stop_pct))
                target_pct = self.ev_model.trade_config.target_pct
                if selected["direction"] == "LONG":
                    stop, target = price * (1 - stop_pct), price * (1 + target_pct)
                else:
                    stop, target = price * (1 + stop_pct), price * (1 - target_pct)
                proposals.append(EntryProposal(
                    symbol=selected["symbol"], direction=selected["direction"], quantity=int(selected["quantity"]),
                    reference_price=price, stop_price=stop, target_price=target,
                    expected_ev_bps=float(selected["net_ev_bps"]), sector=selected.get("sector", "UNKNOWN"),
                    volatility_fraction=float(selected.get("realized_vol_5m", 0.0)),
                    metadata={
                        "alpha_confidence": float(selected["confidence"]),
                        "p_wait": float(selected["p_wait"]), "p_long": float(selected["p_long"]), "p_short": float(selected["p_short"]),
                        "rank": int(selected.get("rank", 0)), "opportunity_score": float(selected.get("opportunity_score", 0.0)),
                        "feature_timestamp": batch.timestamp.isoformat(),
                    },
                ))
            self.last_diagnostics = {
                "timestamp": batch.timestamp.isoformat(), "coverage": batch.coverage,
                "directional_candidates": len(candidates), "positive_ev": len(ranked), "selected": len(proposals),
                "laya_vetoed": 0 if gate_result is None else len(gate_result.get("vetoed", [])),
            }
            return proposals
        except Exception as exc:
            # Fail closed. Shadow/live runtime should never improvise around bad state.
            self.last_diagnostics = {"error": str(exc), "failed_closed": True}
            return []


def load_ev_trades(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_ev_model_from_trades(
    trades_path: str | Path,
    *,
    trade_config: TradeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    ev_config: EVConfig | None = None,
) -> EmpiricalEVModel:
    model = EmpiricalEVModel(
        trade_config=trade_config or TradeConfig(),
        execution_config=execution_config or ExecutionConfig(),
        config=ev_config,
    )
    return model.fit(load_ev_trades(trades_path))
