"""Per-symbol daily features and future-return research labels."""
import numpy as np
import pandas as pd

FEATURES = ["return_5", "return_20", "sma_20_distance", "sma_50_distance", "volatility_20", "volume_ratio_20"]

def build_features(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    df = frame.sort_values(["symbol", "date"]).copy()
    groups = df.groupby("symbol", sort=False)
    close = groups["close"]
    df["return_5"] = close.pct_change(5, fill_method=None)
    df["return_20"] = close.pct_change(20, fill_method=None)
    df["sma_20_distance"] = df["close"] / close.transform(lambda s: s.rolling(20).mean()) - 1
    df["sma_50_distance"] = df["close"] / close.transform(lambda s: s.rolling(50).mean()) - 1
    df["volatility_20"] = groups["close"].transform(lambda s: s.pct_change(fill_method=None).rolling(20).std())
    df["volume_ratio_20"] = df["volume"] / groups["volume"].transform(lambda s: s.rolling(20).mean())
    df["target_5d_return"] = close.shift(-horizon) / df["close"] - 1
    return df.replace([np.inf, -np.inf], np.nan)
