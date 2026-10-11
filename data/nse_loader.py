"""Load user-supplied NSE daily OHLCV; no scraping or API credentials."""
from pathlib import Path
import pandas as pd

REQUIRED = ("date", "symbol", "open", "high", "low", "close", "volume")

def load_daily_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame[list(("open", "high", "low", "close"))].le(0).any().any():
        raise ValueError("OHLC prices must be positive")
    if frame["volume"].lt(0).any():
        raise ValueError("Volume cannot be negative")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date rows")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("High is below another OHLC price")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Low is above another OHLC price")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)
