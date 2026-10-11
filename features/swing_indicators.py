"""Causal daily swing indicators, with explicit warm-up and gap handling."""
import numpy as np
import pandas as pd
from data.feature_inputs import validate_prices


def wilder(values, period):
    """SMA-seeded Wilder average; missing observations reset the seed."""
    result = np.full(len(values), np.nan)
    seed = []
    previous = np.nan
    for position, value in enumerate(values.to_numpy(dtype=float)):
        if not np.isfinite(value):
            seed = []; previous = np.nan
        elif np.isnan(previous):
            seed.append(value)
            if len(seed) == period:
                previous = float(np.mean(seed)); result[position] = previous
        else:
            previous += (value - previous) / period
            result[position] = previous
    return pd.Series(result, index=values.index)


def supertrend(high, low, close, atr, multiplier=3):
    index = close.index
    high, low, close, atr = (series.to_numpy(dtype=float) for series in [high, low, close, atr])
    line = np.full(len(close), np.nan)
    direction = np.full(len(close), np.nan)
    upper = lower = previous_close = np.nan
    trend = -1
    for i in range(len(close)):
        if not np.isfinite([high[i], low[i], close[i], atr[i]]).all():
            upper = lower = previous_close = np.nan
            continue
        basic_upper = (high[i] + low[i]) / 2 + multiplier * atr[i]
        basic_lower = (high[i] + low[i]) / 2 - multiplier * atr[i]
        if np.isnan(upper):
            upper, lower, trend = basic_upper, basic_lower, -1
        else:
            next_upper = basic_upper if basic_upper < upper or previous_close > upper else upper
            next_lower = basic_lower if basic_lower > lower or previous_close < lower else lower
            if trend == -1 and close[i] > next_upper: trend = 1
            elif trend == 1 and close[i] < next_lower: trend = -1
            upper, lower = next_upper, next_lower
        direction[i] = trend
        line[i] = lower if trend == 1 else upper
        previous_close = close[i]
    return pd.Series(line, index=index), pd.Series(direction, index=index)


def _segment(frame):
    c, h, l, v = (frame[name] for name in ['close', 'high', 'low', 'volume'])
    out = pd.DataFrame(index=frame.index)
    for period in [9, 21, 50]:
        out[f'ema_{period}'] = c.ewm(span=period, adjust=False, min_periods=period).mean()
    delta = c.diff()
    gain, loss = wilder(delta.clip(lower=0), 14), wilder(-delta.clip(upper=0), 14)
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask(loss.eq(0) & gain.gt(0), 100).mask(gain.eq(0) & loss.gt(0), 0)
    out['rsi_14'] = rsi.mask(gain.eq(0) & loss.eq(0), 50)
    out['macd'] = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    out['macd_signal'] = out.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out['macd_histogram'] = out.macd - out.macd_signal
    out['volume_ratio_20'] = v / v.rolling(20, min_periods=20).mean().replace(0, np.nan)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out['atr_14'] = wilder(tr, 14)
    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0).mask(up.isna())
    minus = down.where((down > up) & (down > 0), 0).mask(down.isna())
    denominator = wilder(tr.mask(up.isna()), 14)
    p, m = wilder(plus, 14), wilder(minus, 14)
    out['plus_di_14'] = (100*p/denominator.replace(0, np.nan)).mask(denominator.eq(0), 0)
    out['minus_di_14'] = (100*m/denominator.replace(0, np.nan)).mask(denominator.eq(0), 0)
    total = out.plus_di_14 + out.minus_di_14
    dx = (100*(out.plus_di_14-out.minus_di_14).abs()/total.replace(0, np.nan)).mask(total.eq(0), 0)
    out['adx_14'] = wilder(dx, 14)
    out['bb_middle_20'] = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    out['bb_upper_20_2'] = out.bb_middle_20 + 2*sd
    out['bb_lower_20_2'] = out.bb_middle_20 - 2*sd
    out['bb_width'] = 4*sd/out.bb_middle_20
    out['supertrend_10_3'], out['supertrend_direction'] = supertrend(h, l, c, wilder(tr, 10))
    out['high_252'] = h.rolling(252, min_periods=252).max()
    out['high_252_proximity'] = c/out.high_252
    out['high_252_distance_pct'] = 100*(1-out.high_252_proximity)
    return out


def calculate_features(prices, benchmark):
    """Compute before filtering years. Benchmark defines the session calendar."""
    prices = validate_prices(prices)
    benchmark = benchmark.copy()
    benchmark['date'] = pd.to_datetime(benchmark.date, errors='raise')
    if benchmark.date.dt.tz is not None or not benchmark.date.eq(benchmark.date.dt.normalize()).all():
        raise ValueError('Expected daily benchmark session dates')
    benchmark = benchmark.sort_values('date').set_index('date')
    if benchmark.index.has_duplicates or benchmark.index.isna().any() or benchmark.empty:
        raise ValueError('Invalid benchmark dates')
    index_close = pd.to_numeric(benchmark.nifty500_close, errors='raise')
    if not np.isfinite(index_close).all() or index_close.le(0).any():
        raise ValueError('Invalid benchmark closes')
    if not prices.date.isin(benchmark.index).all():
        raise ValueError('Stock date missing from benchmark calendar')
    results = []
    for symbol, stock in prices.groupby('symbol', sort=True):
        calendar = stock.set_index('date').reindex(benchmark.index)
        valid = calendar[['open','high','low','close']].gt(0).all(axis=1)
        valid &= calendar.high.ge(calendar[['open','low','close']].max(axis=1)) & calendar.low.le(calendar[['open','high','close']].min(axis=1))
        # Missing/invalid bars break recursive and rolling indicator history.
        segments = (~valid).cumsum()
        values = pd.DataFrame(index=calendar.index)
        for _, segment in calendar[valid].groupby(segments[valid]):
            computed = _segment(segment)
            values = values.combine_first(computed)
        for period in [20,60]:
            values[f'relative_strength_{period}'] = (calendar.close/calendar.close.shift(period))/(index_close/index_close.shift(period))-1
        observed = calendar.symbol.notna()
        result = calendar.loc[observed].join(values)
        result['features_ready'] = result[list(_segment(calendar.iloc[:0]).columns) + ['relative_strength_20','relative_strength_60']].notna().all(axis=1)
        results.append(result.reset_index())
    return pd.concat(results, ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
