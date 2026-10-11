"""Minimal trade P&L primitive; not a portfolio backtester."""
def net_long_return(entry_price: float, exit_price: float, round_trip_cost_fraction: float = 0.002) -> float:
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Prices must be positive")
    if not 0 <= round_trip_cost_fraction < 1:
        raise ValueError("Invalid cost fraction")
    return exit_price / entry_price - 1 - round_trip_cost_fraction
