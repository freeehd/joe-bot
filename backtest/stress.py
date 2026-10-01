"""Deterministic adverse execution scenarios for robustness testing."""

from __future__ import annotations

from dataclasses import replace

from backtest.engine import ExecutionConfig


def execution_stress_scenarios(base: ExecutionConfig) -> dict[str, ExecutionConfig]:
    """Return baseline plus deliberately harsher execution assumptions."""

    return {
        "baseline": base,
        "double_spread": replace(base, spread_bps=base.spread_bps * 2.0),
        "double_slippage": replace(base, slippage_bps=base.slippage_bps * 2.0),
        "extra_delay": replace(base, entry_delay_bars=base.entry_delay_bars + 1),
        "combined_adverse": replace(
            base,
            spread_bps=base.spread_bps * 2.0,
            slippage_bps=base.slippage_bps * 2.0,
            entry_delay_bars=base.entry_delay_bars + 1,
        ),
    }
