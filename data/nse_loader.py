"""Canonical, validated daily NSE OHLCV data; timestamps are local session dates."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ("date", "symbol", "open", "high", "low", "close", "volume")


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED].copy()
    if frame.empty:
        raise ValueError("No daily OHLCV rows")
    if frame[["date", "symbol"]].isna().any().any():
        raise ValueError("Missing date or symbol")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.date.isna().any() or frame.date.dt.tz is not None:
        raise ValueError("Dates must be timezone-naive session dates")
    if not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Expected daily session dates, not intraday timestamps")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if frame.symbol.eq("").any():
        raise ValueError("Empty symbol")
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV must be finite and non-missing")
    if frame[["open", "high", "low", "close"]].le(0).any().any():
        raise ValueError("OHLC prices must be positive")
    if frame.volume.lt(0).any():
        raise ValueError("Volume cannot be negative")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date rows")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("High is below another OHLC price")
    if (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Low is above another OHLC price")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_daily_csv(path: str | Path) -> pd.DataFrame:
    return validate_daily(pd.read_csv(path))
