# Daily swing indicators: 2019–2022

Run from the repository root in VS Code or a terminal:

```sh
python -m data.feature_inputs --workers 16
python -m features.generate_features
python -m pytest -q
```

`derived/features_2019.csv.xz` through `derived/features_2022.csv.xz` are separate yearly feature files. Read them directly with pandas. Computation uses the complete input history before splitting years, so indicators do not reset in January. The manifest records source checksums, schema, coverage, missing values, and ready-row counts. Files include daily OHLCV for auditability and contain no forward labels.

| Indicator | Columns and convention |
|---|---|
| EMA | `ema_9`, `ema_21`, `ema_50`; alpha 2/(n+1), first-close seed, minimum n observations |
| RSI | `rsi_14`; SMA-seeded Wilder gains/losses, flat=50, gains only=100, losses only=0 |
| MACD | `macd`, `macd_signal`, `macd_histogram`; EMA12−EMA26, EMA9 signal, first valid MACD seed |
| Volume ratio | `volume_ratio_20`; current volume / 20-session mean including current session; all-zero mean missing |
| ADX | `adx_14`, `plus_di_14`, `minus_di_14`; SMA-seeded Wilder smoothing, 14 differences then 14 DX values; first ADX on bar 28 |
| Bollinger Bands | `bb_middle_20`, `bb_upper_20_2`, `bb_lower_20_2`, `bb_width`; SMA20 ± 2 population SD; width=(upper−lower)/middle |
| ATR | `atr_14`; Wilder true range, first bar high−low, first value on bar 14 |
| Supertrend | `supertrend_10_3`, `supertrend_direction`; Wilder ATR10 ×3, final trailing bands, initially bearish; +1 bullish, −1 bearish |
| Market relative strength | `relative_strength_20`, `relative_strength_60`; stock gross return / Nifty 500 gross return −1, positive means outperformance |
| 52-week high proximity | `high_252`, `high_252_proximity`, `high_252_distance_pct`; rolling 252-session daily high, close/high, 100×(1−close/high) |

All values are available after the specified session's close. Recursive and rolling indicators reset after an absent or invalid stock bar; no forward/backward filling is used. Relative-strength endpoints are separated by exactly 20/60 benchmark sessions, rather than counting a suspended stock's observed rows. It requires both stock endpoints but permits missing interior observations. `features_ready` is true only when all requested indicator columns are present; warm-up rows remain in the files with empty fields. Newly listed stocks need enough history to qualify. The 52-week measure is a 252-session approximation, with at least 252 contiguous valid bars.

Supporting daily highs/lows and index closes are in `data/feature_inputs/`. History starts December 2017 to warm up full-calendar 2019 output. The original open/close/volume dataset remains intact. The supporting collector reconciles overlapping open/close/volume with that dataset and audits missing archive dates.

Prices are raw, unadjusted NSE prices and can jump on splits, bonuses, or other corporate actions. The universe is today's constituent snapshot, not historical membership; renamed, delisted, and not-yet-listed securities have incomplete coverage. These are research feature files, not a corporate-action-cleaned training dataset. The Nifty 500 benchmark is the price index, not its total-return variant. Live trading remains disabled.

Formula references: [TradingView RSI](https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/), [DMI](https://www.tradingview.com/support/solutions/43000502250-directional-movement-dmi/), [Supertrend](https://www.tradingview.com/support/solutions/43000634738-supertrend/). Explicit seed and missing-data conventions above govern this implementation.
