"""Features known at signal close; next-open to future-open execution labels."""
import numpy as np
import pandas as pd
from data.nse_loader import validate_daily

FEATURES = ["return_5", "return_20", "sma_20_distance", "sma_50_distance",
            "volatility_20", "volume_ratio_20", "rsi_14", "atr_14_fraction"]
TARGET = "target_return"


def build_features(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    df = validate_daily(frame)
    groups = df.groupby("symbol", sort=False)
    close = groups["close"]
    df["return_5"] = close.pct_change(5, fill_method=None)
    df["return_20"] = close.pct_change(20, fill_method=None)
    for window in (20, 50):
        df[f"sma_{window}_distance"] = df.close / close.transform(lambda s: s.rolling(window).mean()) - 1
    df["volatility_20"] = close.transform(lambda s: s.pct_change(fill_method=None).rolling(20).std())
    df["volume_ratio_20"] = df.volume / groups.volume.transform(lambda s: s.rolling(20).mean())
    delta = close.diff()
    gains = delta.clip(lower=0).groupby(df.symbol).transform(lambda s: s.rolling(14).mean())
    losses = (-delta.clip(upper=0)).groupby(df.symbol).transform(lambda s: s.rolling(14).mean())
    df["rsi_14"] = 100 * gains / (gains + losses)
    df.loc[(gains + losses).eq(0), "rsi_14"] = 50
    previous = close.shift(1)
    true_range = pd.concat([df.high - df.low, (df.high - previous).abs(),
                            (df.low - previous).abs()], axis=1).max(axis=1)
    df["atr_14_fraction"] = true_range.groupby(df.symbol).transform(lambda s: s.rolling(14).mean()) / df.close
    # Calculate dates BEFORE dropping feature/label rows, preserving exact alignment.
    df["entry_date"] = groups.date.shift(-1)
    df["label_end_date"] = groups.date.shift(-(horizon + 1))
    df["entry_price"] = groups.open.shift(-1)
    df["exit_price"] = groups.open.shift(-(horizon + 1))
    df[TARGET] = df.exit_price / df.entry_price - 1
    # Backward-compatible name, now explicitly execution-aligned for the chosen horizon.
    df["target_5d_return"] = df[TARGET]
    return df.replace([np.inf, -np.inf], np.nan)
