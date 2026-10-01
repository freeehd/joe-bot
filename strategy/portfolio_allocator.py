"""V0.7 expected-value ranking and risk/correlation-aware capital allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor

import pandas as pd

from backtest.engine import TradeConfig
from market.sectors import sector_for
from risk.portfolio_risk import candidate_correlation_penalty, directional_correlation
from strategy.ev_model import EmpiricalEVModel


@dataclass(frozen=True)
class PortfolioConstraints:
    max_positions: int = 3
    max_deployed_fraction: float = 0.70
    max_position_fraction: float = 0.30
    risk_per_trade_fraction: float = 0.0025
    max_total_risk_fraction: float = 0.01
    max_sector_deployed_fraction: float = 0.40
    max_long_deployed_fraction: float = 0.70
    max_short_deployed_fraction: float = 0.40
    max_correlated_risk_fraction: float = 0.005
    correlation_threshold: float = 0.75
    min_position_dollars: float = 20.0
    min_net_ev_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        fraction_names = (
            "max_deployed_fraction",
            "max_position_fraction",
            "risk_per_trade_fraction",
            "max_total_risk_fraction",
            "max_sector_deployed_fraction",
            "max_long_deployed_fraction",
            "max_short_deployed_fraction",
            "max_correlated_risk_fraction",
        )
        for name in fraction_names:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.min_position_dollars < 0:
            raise ValueError("min_position_dollars must be non-negative")


class EVRanker:
    """Rank directional candidates by estimated net EV per unit of stop risk."""

    def __init__(self, ev_model: EmpiricalEVModel) -> None:
        self.ev_model = ev_model

    def rank(
        self,
        candidates: list[dict],
        *,
        correlations: pd.DataFrame | None = None,
        current_positions: list[dict] | None = None,
        min_net_ev_bps: float = 0.0,
    ) -> list[dict]:
        positions = current_positions or []
        ranked: list[dict] = []
        stop_pct = self.ev_model.trade_config.stop_pct

        for source in candidates:
            direction = str(source.get("direction", "WAIT"))
            if direction not in {"LONG", "SHORT"}:
                continue
            p_wait = float(source["p_wait"])
            p_long = float(source["p_long"])
            p_short = float(source["p_short"])
            confidence = float(source.get("confidence", max(p_long, p_short, p_wait)))
            estimate = self.ev_model.estimate(
                side=direction,  # type: ignore[arg-type]
                p_wait=p_wait,
                p_long=p_long,
                p_short=p_short,
                confidence=confidence,
            )
            if estimate.net_ev_bps <= min_net_ev_bps:
                continue

            candidate = source.copy()
            symbol = str(candidate["symbol"])
            multiplier, max_corr = candidate_correlation_penalty(
                symbol=symbol,
                side=direction,
                positions=positions,
                correlations=correlations,
            )
            risk_adjusted_ev = estimate.blended_net_ev / stop_pct
            # Confidence is a mild tiebreaker; net EV/risk remains dominant.
            confidence_factor = 0.5 + 0.5 * confidence
            base_score = risk_adjusted_ev * confidence_factor
            score = base_score * multiplier

            candidate.update(
                {
                    "sector": candidate.get("sector") or sector_for(symbol),
                    "stop_pct": float(candidate.get("stop_pct", stop_pct)),
                    "net_ev": estimate.blended_net_ev,
                    "net_ev_bps": estimate.net_ev_bps,
                    "structural_net_ev": estimate.structural_net_ev,
                    "empirical_net_ev": estimate.empirical_net_ev,
                    "empirical_samples": estimate.empirical_samples,
                    "empirical_weight": estimate.empirical_weight,
                    "correlation_multiplier": multiplier,
                    "max_directional_correlation": max_corr,
                    "base_opportunity_score": float(base_score),
                    "opportunity_score": float(score),
                }
            )
            ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                item["opportunity_score"],
                item["net_ev_bps"],
                item.get("confidence", 0.0),
            ),
            reverse=True,
        )
        for index, candidate in enumerate(ranked, start=1):
            candidate["rank"] = index
        return ranked


class PortfolioAllocatorV2:
    """Risk-based allocator with hard portfolio exposure constraints.

    This is intentionally *not* Kelly sizing. Each candidate begins with a
    fixed maximum account-risk budget and is then reduced by notional, total
    risk, sector, direction, and correlation limits. Capital may remain idle.
    """

    def __init__(
        self,
        *,
        constraints: PortfolioConstraints | None = None,
        trade_config: TradeConfig | None = None,
    ) -> None:
        self.constraints = constraints or PortfolioConstraints()
        self.trade_config = trade_config or TradeConfig()

    @staticmethod
    def _exposure(positions: list[dict], direction: str | None = None) -> float:
        return float(
            sum(
                float(position.get("allocation", 0.0))
                for position in positions
                if direction is None or position.get("direction") == direction
            )
        )

    @staticmethod
    def _risk(positions: list[dict]) -> float:
        return float(sum(float(position.get("risk_dollars", 0.0)) for position in positions))

    @staticmethod
    def _sector_exposure(positions: list[dict], sector: str) -> float:
        return float(
            sum(
                float(position.get("allocation", 0.0))
                for position in positions
                if str(position.get("sector", "UNKNOWN")) == sector
            )
        )

    def _correlated_risk_room(
        self,
        *,
        candidate: dict,
        positions: list[dict],
        correlations: pd.DataFrame | None,
        account_equity: float,
    ) -> float:
        limit = account_equity * self.constraints.max_correlated_risk_fraction
        correlated_risk = 0.0
        symbol = str(candidate["symbol"])
        side = str(candidate["direction"])

        for position in positions:
            other = str(position["symbol"])
            if symbol == other:
                return 0.0
            corr = 0.0
            if correlations is not None and not correlations.empty:
                if symbol in correlations.index and other in correlations.columns:
                    corr = float(correlations.loc[symbol, other])
            pnl_corr = directional_correlation(corr, side, str(position["direction"]))
            if pnl_corr >= self.constraints.correlation_threshold:
                correlated_risk += float(position.get("risk_dollars", 0.0))
        return max(0.0, limit - correlated_risk)

    def allocate(
        self,
        ranked_candidates: list[dict],
        *,
        account_equity: float,
        correlations: pd.DataFrame | None = None,
        current_positions: list[dict] | None = None,
    ) -> dict:
        if account_equity <= 0:
            raise ValueError("account_equity must be > 0")

        current = [position.copy() for position in (current_positions or [])]
        selected: list[dict] = []
        rejected: list[dict] = []
        all_positions = current.copy()

        c = self.constraints
        max_total_notional = account_equity * c.max_deployed_fraction
        max_position_notional = account_equity * c.max_position_fraction
        max_total_risk = account_equity * c.max_total_risk_fraction
        max_sector_notional = account_equity * c.max_sector_deployed_fraction
        max_long_notional = account_equity * c.max_long_deployed_fraction
        max_short_notional = account_equity * c.max_short_deployed_fraction
        per_trade_risk = account_equity * c.risk_per_trade_fraction

        def reject(candidate: dict, reason: str) -> None:
            rejected.append(
                {
                    "symbol": candidate.get("symbol"),
                    "direction": candidate.get("direction"),
                    "reason": reason,
                    "net_ev_bps": candidate.get("net_ev_bps"),
                }
            )

        remaining = [candidate.copy() for candidate in ranked_candidates]

        while remaining:
            if len(all_positions) >= c.max_positions:
                for leftover in remaining:
                    reject(leftover, "maximum positions reached")
                break

            # Re-rank greedily after every accepted position so a slightly
            # weaker but genuinely diversifying idea can outrank a clustered
            # second bet once the first position is already in the portfolio.
            rescored: list[tuple[float, dict]] = []
            for candidate in remaining:
                multiplier, max_corr = candidate_correlation_penalty(
                    symbol=str(candidate["symbol"]),
                    side=str(candidate["direction"]),
                    positions=all_positions,
                    correlations=correlations,
                )
                base_score = float(
                    candidate.get(
                        "base_opportunity_score",
                        candidate.get("opportunity_score", 0.0),
                    )
                )
                candidate["correlation_multiplier"] = multiplier
                candidate["max_directional_correlation"] = max_corr
                candidate["selection_score"] = base_score * multiplier
                rescored.append((candidate["selection_score"], candidate))

            rescored.sort(
                key=lambda item: (
                    item[0],
                    float(item[1].get("net_ev_bps", 0.0)),
                    float(item[1].get("confidence", 0.0)),
                ),
                reverse=True,
            )
            candidate = rescored[0][1]
            remaining.remove(candidate)

            if float(candidate.get("net_ev_bps", float("-inf"))) <= c.min_net_ev_bps:
                reject(candidate, "net EV below threshold")
                continue
            if str(candidate.get("direction")) not in {"LONG", "SHORT"}:
                reject(candidate, "non-directional candidate")
                continue
            if any(str(position.get("symbol")) == str(candidate.get("symbol")) for position in all_positions):
                reject(candidate, "symbol already held")
                continue

            price = float(candidate.get("price", 0.0))
            stop_pct = float(candidate.get("stop_pct", self.trade_config.stop_pct))
            if price <= 0 or stop_pct <= 0:
                reject(candidate, "invalid price/stop distance")
                continue

            sector = str(candidate.get("sector") or sector_for(str(candidate["symbol"])))
            direction = str(candidate["direction"])
            risk_per_share = price * stop_pct
            total_risk_room = max(0.0, max_total_risk - self._risk(all_positions))
            correlated_risk_room = self._correlated_risk_room(
                candidate=candidate,
                positions=all_positions,
                correlations=correlations,
                account_equity=account_equity,
            )
            risk_budget = min(per_trade_risk, total_risk_room, correlated_risk_room)
            if risk_budget <= 0:
                reject(candidate, "portfolio/correlation risk budget exhausted")
                continue

            risk_qty = floor(risk_budget / risk_per_share)
            if risk_qty < 1:
                reject(candidate, "risk budget smaller than one share")
                continue

            notional_rooms = [
                max_position_notional,
                max(0.0, max_total_notional - self._exposure(all_positions)),
                max(0.0, max_sector_notional - self._sector_exposure(all_positions, sector)),
            ]
            if direction == "LONG":
                notional_rooms.append(max(0.0, max_long_notional - self._exposure(all_positions, "LONG")))
            else:
                notional_rooms.append(max(0.0, max_short_notional - self._exposure(all_positions, "SHORT")))

            notional_cap = min(notional_rooms)
            notional_qty = floor(notional_cap / price)
            quantity = min(risk_qty, notional_qty)
            if quantity < 1:
                reject(candidate, "notional/exposure budget exhausted")
                continue

            allocation = float(quantity * price)
            if allocation < c.min_position_dollars:
                reject(candidate, "position below minimum dollars")
                continue
            actual_risk = float(quantity * risk_per_share)

            position = candidate.copy()
            position.update(
                {
                    "sector": sector,
                    "quantity": int(quantity),
                    "allocation": allocation,
                    "risk_dollars": actual_risk,
                    "risk_fraction": actual_risk / account_equity,
                    "stop_distance_dollars": risk_per_share,
                }
            )
            selected.append(position)
            all_positions.append(position)

        deployed = self._exposure(all_positions)
        selected_deployed = self._exposure(selected)
        total_risk = self._risk(all_positions)

        return {
            "positions": selected,
            "rejected": rejected,
            "selected_count": len(selected),
            "selected_capital": selected_deployed,
            "capital_deployed": deployed,
            "cash_available": max(0.0, account_equity - deployed),
            "portfolio_risk_dollars": total_risk,
            "portfolio_risk_fraction": total_risk / account_equity,
            "long_exposure": self._exposure(all_positions, "LONG"),
            "short_exposure": self._exposure(all_positions, "SHORT"),
            "constraints": asdict(c),
        }
