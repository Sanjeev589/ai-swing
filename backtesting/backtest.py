"""Minimal trade P&L primitive; not a portfolio backtester."""
import math

def net_long_return(entry_price: float, exit_price: float, round_trip_cost_fraction: float = 0.002) -> float:
    if not all(math.isfinite(p) and p > 0 for p in (entry_price, exit_price)):
        raise ValueError("Prices must be positive")
    if not 0 <= round_trip_cost_fraction < 1:
        raise ValueError("Invalid cost fraction")
    return exit_price / entry_price - 1 - round_trip_cost_fraction
