"""Trading utilities package."""

from .simulator import (
    Bar,
    SimulationConfig,
    PortfolioSimulator,
    simulate_trades_with_tp_sl,
    simulate_trades_with_tp_sl_more_aggressive,
    print_portfolio_report,
    class_to_raw,
)

__all__ = [
    "Bar",
    "SimulationConfig",
    "PortfolioSimulator",
    "simulate_trades_with_tp_sl",
    "simulate_trades_with_tp_sl_more_aggressive",
    "print_portfolio_report",
    "class_to_raw",
]
