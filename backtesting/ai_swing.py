"""Research-only Nifty 500 swing system: causal setups, purged LightGBM, risk-managed trades.

No broker integration or live order placement. All trading decisions are made after the
signal close and trades start no earlier than the following session's opening price.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingResearchConfig:
    test_start: str = '2022-01-01'
    test_end: str = '2022-12-31'
    horizon: int = 10
    min_train_sessions: int = 252
    refit_sessions: int = 63
    estimators: int = 80
    min_breadth_stocks: int = 50
    min_breadth_fraction: float = .45
    min_turnover_inr: float = 50_000_000
    max_positions: int = 10
    max_position_fraction: float = .15
    max_gross_exposure: float = .75
    risk_per_trade: float = .005
    atr_stop_multiplier: float = 2.0
    max_hold_sessions: int = 15
    cost_bps_per_side: float = 25.0
    initial_capital: float = 1_000_000
    seed: int = 42

    def __post_init__(self):
        for key in ('horizon', 'min_train_sessions', 'refit_sessions', 'estimators',
                    'min_breadth_stocks', 'max_positions', 'max_hold_sessions'):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'{key} must be a positive integer')
        if pd.Timestamp(self.test_start) >= pd.Timestamp(self.test_end):
            raise ValueError('test_start must precede test_end')
        for key in ('min_breadth_fraction', 'max_position_fraction',
                    'max_gross_exposure', 'risk_per_trade'):
            v = getattr(self, key)
            if not np.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f'{key} must be in (0,1]')
        for key in ('min_turnover_inr', 'cost_bps_per_side'):
            v = getattr(self, key)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.cost_bps_per_side >= 10000 or self.atr_stop_multiplier <= 0 or self.initial_capital <= 0:
            raise ValueError('Invalid costs, stop multiplier or initial capital')


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Accept one OHLCV CSV or a directory of dated OHLCV archives; never infer highs/lows."""
    from data.nse_loader import validate_daily
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.glob('ohlcv_*.csv.gz'))
    elif path.is_file():
        paths = [path]
    else:
        paths = []
    if not paths:
        raise ValueError('No OHLCV files found (open/close/volume-only data is insufficient)')
    return validate_daily(pd.concat((pd.read_csv(p, parse_dates=['date']) for p in paths),
                                    ignore_index=True))


def prepare_candidates(features: pd.DataFrame, benchmark: pd.DataFrame,
                       cfg: SwingResearchConfig, sectors: pd.DataFrame | None = None) -> pd.DataFrame:
    """All candidate features and regime decisions use data known by signal close."""
    from models.swing_dataset import model_inputs
    df = model_inputs(features).sort_values(['symbol', 'date']).copy()
    g = df.groupby('symbol', sort=False)
    df['prior_high_20'] = g.high.transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    df['prev_close'] = g.close.shift(1)
    df['turnover_20'] = (df.close * df.volume).groupby(df.symbol).transform(
        lambda s: s.rolling(20, min_periods=20).mean())
    # Block potential unadjusted splits/bonuses and other extreme discontinuities.
    df['large_jump_recent'] = (df.close.div(df.prev_close).sub(1).abs().gt(.25)
                               .groupby(df.symbol).transform(lambda s: s.rolling(20, min_periods=1).max())
                               .fillna(True).astype(bool))
    df['atr_fraction'] = df.atr_14 / df.close
    uptrend = df.ema_21.gt(df.ema_50) & df.close.gt(df.ema_21)
    df['breakout'] = (uptrend & df.close.gt(df.prior_high_20)
                      & df.volume_ratio_20.ge(1.2) & df.adx_14.ge(18)
                      & df.relative_strength_20.gt(0) & df.high_252_proximity.ge(.85))
    df['pullback'] = (uptrend & df.low.le(df.ema_21 * 1.01)
                      & df.close.gt(df.prev_close) & df.rsi_14.between(45, 75)
                      & df.relative_strength_60.gt(0) & df.high_252_proximity.ge(.80))
    df['setup'] = np.select([df.breakout, df.pullback], ['breakout', 'pullback'], default='none')
    df['liquid'] = (df.turnover_20.ge(cfg.min_turnover_inr)
                    & df.atr_fraction.between(.005, .10)
                    & ~df.large_jump_recent)
    b = benchmark.copy()
    if not {'date', 'nifty500_close'}.issubset(b):
        raise ValueError('Benchmark needs date,nifty500_close')
    b['date'] = pd.to_datetime(b.date)
    if b.date.dt.tz is not None or not b.date.eq(b.date.dt.normalize()).all():
        raise ValueError('Benchmark must use daily timezone-naive dates')
    if b.date.duplicated().any() or not b.date.is_monotonic_increasing:
        b = b.sort_values('date')
        if b.date.duplicated().any():
            raise ValueError('Duplicate benchmark sessions')
    if not np.isfinite(b.nifty500_close).all() or b.nifty500_close.le(0).any():
        raise ValueError('Invalid benchmark prices')
    b['sma_200'] = b.nifty500_close.rolling(200, min_periods=200).mean()
    b['momentum_20'] = b.nifty500_close.pct_change(20, fill_method=None)
    breadth = df.assign(above_50=df.close.gt(df.ema_50) & df.ema_50.notna()).groupby('date').agg(
        breadth_count=('above_50', 'size'), breadth_fraction=('above_50', 'mean')).reset_index()
    b = b.merge(breadth, on='date', how='left', validate='one_to_one')
    b['market_ok'] = (b.nifty500_close.gt(b.sma_200) & b.momentum_20.gt(0)
                      & b.breadth_count.ge(cfg.min_breadth_stocks)
                      & b.breadth_fraction.ge(cfg.min_breadth_fraction))
    df = df.merge(b[['date', 'market_ok', 'breadth_fraction', 'momentum_20']],
                  on='date', how='left', validate='many_to_one')
    df['market_ok'] = df.market_ok.fillna(False).astype(bool)
    df['sector_strength_20'] = np.nan
    if sectors is not None:
        if not {'symbol', 'sector'}.issubset(sectors):
            raise ValueError('sector CSV requires symbol,sector')
        s = sectors[['symbol', 'sector']].drop_duplicates()
        if s.symbol.duplicated().any():
            raise ValueError('Conflicting sector membership')
        df = df.merge(s, on='symbol', how='left', validate='many_to_one')
        # Optional sector filter uses only available same-day cross-sectional returns.
        df['sector_strength_20'] = df.groupby(['date', 'sector']).relative_strength_20.transform('median')
    else:
        df['sector'] = pd.NA
    df['eligible'] = (df.model_ready & df.market_ok & df.liquid &
                      df.setup.ne('none') & df.atr_14.gt(0))
    if sectors is not None:
        df['eligible'] &= df.sector_strength_20.gt(0)
    return df.sort_values(['date', 'symbol']).reset_index(drop=True)


def purged_predictions(frame: pd.DataFrame, cfg: SwingResearchConfig):
    """Train a fresh ranker every N test sessions with strictly past realized labels."""
    from models.swing_dataset import MODEL_FEATURES
    from models.train_swing_ranker import new_ranker, ranking_query
    dates = pd.DatetimeIndex(sorted(frame.loc[
        frame.date.between(cfg.test_start, cfg.test_end), 'date'].unique()))
    if len(dates) < 2:
        raise ValueError('Insufficient test sessions')
    output, folds = [], []
    for offset in range(0, len(dates), cfg.refit_sessions):
        start = dates[offset]
        test_dates = dates[offset:offset + cfg.refit_sessions]
        hist = frame[(frame.date.lt(start)) & (frame.label_end_date.lt(start)) &
                     frame.model_ready & frame.target_return.notna()]
        if hist.date.nunique() < cfg.min_train_sessions:
            raise ValueError(f'Insufficient purged training sessions before {start.date()}')
        fit, labels, groups = ranking_query(hist)
        ranker = new_ranker(cfg.estimators, cfg.seed)
        ranker.fit(fit[MODEL_FEATURES], labels, group=groups)
        candidates = frame[frame.date.isin(test_dates) & frame.eligible].copy()
        if not candidates.empty:
            candidates['ai_score'] = ranker.predict(candidates[MODEL_FEATURES])
            # Same eligibility, different rank policy: isolates the AI contribution.
            candidates['rule_score'] = (candidates.relative_strength_20 +
                                        candidates.relative_strength_60 * .5 +
                                        candidates.breakout.astype(float) * .025)
            candidates['fold'] = len(folds)
            output.append(candidates)
        folds.append({'fold': len(folds), 'test_start': start,
                      'test_end': test_dates[-1], 'train_start': fit.date.min(),
                      'train_end': fit.date.max(), 'last_training_label': fit.label_end_date.max(),
                      'train_sessions': fit.date.nunique(), 'train_rows': len(fit),
                      'candidate_rows': len(candidates)})
    if output:
        scored = pd.concat(output, ignore_index=True)
    else:
        scored = frame.iloc[:0].copy().assign(ai_score=pd.Series(dtype=float),
                                               rule_score=pd.Series(dtype=float),
                                               fold=pd.Series(dtype=int))
    return scored, pd.DataFrame(folds)


def simulate(prices: pd.DataFrame, scored: pd.DataFrame, cfg: SwingResearchConfig,
             ranking: str = 'ai_score'):
    """Long-only delivery shares; next-open entries, gap-aware daily stops, close-marked equity.

    A stop breached below opening price fills at that opening price; otherwise the
    stop level is assumed filled when daily low breaches it. Intraday execution is
    still an approximation. Stops trail only AFTER the session closes.
    """
    if ranking not in ('ai_score', 'rule_score'):
        raise ValueError('Invalid ranking policy')
    if scored.duplicated(['date', 'symbol']).any():
        raise ValueError('Duplicate signal keys')
    bars = prices[prices.date.between(cfg.test_start, cfg.test_end)].copy()
    if bars.empty:
        raise ValueError('No prices in evaluation window')
    calendar = pd.DatetimeIndex(sorted(bars.date.unique()))
    sessions = {d: g.set_index('symbol') for d, g in bars.groupby('date', sort=True)}
    candidates = {}
    if not scored.empty:
        for date, g in scored.groupby('date'):
            if g[ranking].isna().any():
                raise ValueError('Missing scores')
            candidates[date] = g.sort_values([ranking, 'symbol'], ascending=[False, True])
    cash, holdings, trades, curve, rejected = float(cfg.initial_capital), {}, [], [], []
    cost = cfg.cost_bps_per_side / 10000

    def exit_position(symbol, day, px, reason):
        nonlocal cash
        pos = holdings.pop(symbol)
        proceeds = pos['shares'] * px * (1 - cost)
        cash += proceeds
        invested = pos['shares'] * pos['entry_price'] * (1 + cost)
        trades.append({'symbol': symbol, 'setup': pos['setup'],
                       'signal_date': pos['signal_date'], 'entry_date': pos['entry_date'],
                       'exit_date': day, 'entry_price': pos['entry_price'],
                       'exit_price': float(px), 'shares': pos['shares'],
                       'net_pnl_inr': proceeds - invested,
                       'net_return': proceeds / invested - 1, 'exit_reason': reason})

    for i, day in enumerate(calendar):
        today = sessions[day]
        # A price suspension/delisting during an open position cannot be safely simulated.
        absent = set(holdings).difference(today.index)
        if absent:
            raise ValueError(f'Unpriced open positions on {day.date()}: {sorted(absent)}')
        # Stops based on yesterday's closes, and age/time exits, are applied before entries.
        for symbol in list(holdings):
            pos, bar = holdings[symbol], today.loc[symbol]
            if pos['age'] >= cfg.max_hold_sessions:
                exit_position(symbol, day, bar.open, 'max_hold_open')
            elif bar.open <= pos['stop']:
                exit_position(symbol, day, bar.open, 'gap_stop')
        if i:
            previous = calendar[i-1]
            for _, item in candidates.get(previous, pd.DataFrame()).iterrows():
                symbol = item.symbol
                if len(holdings) >= cfg.max_positions:
                    break
                if symbol in holdings or symbol not in today.index:
                    continue
                bar = today.loc[symbol]
                entry, atr = float(bar.open), float(item.atr_14)
                if not np.isfinite([entry, atr]).all() or atr <= 0:
                    continue
                # Reject large opening gaps after signals; no fill at yesterday's close.
                if entry > float(item.close) + 2 * atr:
                    rejected.append({'date': day, 'symbol': symbol, 'reason': 'large_opening_gap'})
                    continue
                equity_open = cash + sum(p['shares'] * today.loc[s, 'open'] for s, p in holdings.items())
                exposure_open = sum(p['shares'] * today.loc[s, 'open'] for s, p in holdings.items())
                remaining_exposure = max(0., cfg.max_gross_exposure * equity_open - exposure_open)
                cap_budget = min(cash, cfg.max_position_fraction * equity_open, remaining_exposure)
                risk_budget = cfg.risk_per_trade * equity_open
                stop_distance = max(cfg.atr_stop_multiplier * atr, entry * .01)
                shares = int(min(cap_budget / (entry * (1 + cost)),
                                 risk_budget / (stop_distance + 2 * entry * cost)))
                if shares < 1:
                    continue
                outlay = shares * entry * (1 + cost)
                if outlay > cash + 1e-6:
                    raise AssertionError('Risk sizing overspent cash')
                cash -= outlay
                holdings[symbol] = {'shares': shares, 'entry_price': entry, 'entry_date': day,
                                    'signal_date': previous, 'setup': item.setup,
                                    'stop': entry - stop_distance, 'age': 0, 'atr': atr}
        # Today's OHLC path can trigger existing or newly opened hard stops.
        for symbol in list(holdings):
            pos, bar = holdings[symbol], today.loc[symbol]
            if bar.low <= pos['stop']:
                exit_position(symbol, day, min(float(bar.open), pos['stop']), 'intraday_stop')
        # Mark to market at close; trailing stops adjust for the NEXT trading day only.
        equity_close = cash + sum(p['shares'] * today.loc[s, 'close'] for s, p in holdings.items())
        curve.append({'date': day, 'equity': equity_close, 'cash': cash, 'positions': len(holdings)})
        for symbol, pos in holdings.items():
            bar = today.loc[symbol]
            pos['age'] += 1
            # Use the SIGNAL-DAY ATR as a stable, observable trailing-stop distance.
            # Updated here, never retroactively applied to today's lows.
            pos['stop'] = max(pos['stop'], float(bar.close) - cfg.atr_stop_multiplier * pos['atr'])
    # End-of-window liquidation is explicitly modeled at final CLOSE, not an alleged next-open fill.
    if holdings:
        last = calendar[-1]
        for symbol in list(holdings):
            exit_position(symbol, last, sessions[last].loc[symbol, 'close'], 'end_of_sample_close')
        curve[-1]['equity'] = cash
        curve[-1]['cash'] = cash
        curve[-1]['positions'] = 0
    equity = pd.DataFrame(curve)
    equity['daily_return'] = equity.equity.pct_change().fillna(equity.equity.iloc[0] / cfg.initial_capital - 1)
    drawdowns = equity.equity / np.maximum.accumulate(np.r_[cfg.initial_capital, equity.equity.to_numpy()])[1:] - 1
    trades = pd.DataFrame(trades, columns=['symbol','setup','signal_date','entry_date','exit_date',
                                           'entry_price','exit_price','shares','net_pnl_inr','net_return','exit_reason'])
    wins = int(trades.net_pnl_inr.gt(0).sum()) if len(trades) else 0
    gross_wins = float(trades.loc[trades.net_pnl_inr.gt(0), 'net_pnl_inr'].sum())
    gross_losses = float(-trades.loc[trades.net_pnl_inr.lt(0), 'net_pnl_inr'].sum())
    sd = equity.daily_return.std(ddof=1)
    summary = {'ranking': ranking, 'sessions': len(calendar), 'completed_trades': len(trades),
               'winning_trades': wins, 'losing_trades': int(trades.net_pnl_inr.lt(0).sum()) if len(trades) else 0,
               'win_rate': wins / len(trades) if len(trades) else None,
               'profit_factor': gross_wins / gross_losses if gross_losses else None,
               'net_profit_inr': cash - cfg.initial_capital,
               'total_net_return': cash / cfg.initial_capital - 1,
               'max_drawdown': float(drawdowns.min()),
               'sharpe_252': float(np.sqrt(252) * equity.daily_return.mean() / sd) if len(equity) > 1 and sd > 0 else None,
               'rejected_opening_gaps': len(rejected), 'open_positions': 0,
               'paper_only': True}
    return equity, trades, summary


def run(prices_path, benchmark_path, output, cfg=None, price_policy=None, sectors_path=None):
    """Offline end-to-end evaluation with exact input provenance declarations."""
    cfg = cfg or SwingResearchConfig()
    if price_policy not in ('verified-adjusted', 'research-unadjusted'):
        raise ValueError('Explicit --price-policy required: verified-adjusted or research-unadjusted')
    price_root = Path(prices_path)
    manifest = price_root / 'manifest.json' if price_root.is_dir() else price_root.parent / 'manifest.json'
    if price_policy == 'verified-adjusted' and manifest.exists():
        policy = str(json.loads(manifest.read_text()).get('price_adjustment', '')).lower()
        if policy in ('none', 'as-published nse open/close; not adjusted for corporate actions') or 'not adjusted' in policy:
            raise ValueError('Source manifest declares UNADJUSTED prices; use research-unadjusted only')
    prices = load_ohlcv(prices_path)
    benchmark = pd.read_csv(benchmark_path, parse_dates=['date']).sort_values('date')
    if not prices.date.isin(benchmark.date).all():
        raise ValueError('Stock session date absent from Nifty 500 benchmark calendar')
    from features.swing_indicators import calculate_features
    from models.swing_dataset import execution_labels
    features = calculate_features(prices, benchmark)
    sectors = pd.read_csv(sectors_path) if sectors_path else None
    prepared = prepare_candidates(features, benchmark, cfg, sectors=sectors)
    labels = execution_labels(prices, benchmark.date, cfg.horizon)
    prepared = prepared.merge(labels[['date','symbol','label_end_date','target_return']],
                              on=['date','symbol'], how='left', validate='one_to_one')
    scored, folds = purged_predictions(prepared, cfg)
    outputs = {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for strategy in ('ai_score', 'rule_score'):
        equity, trades, summary = simulate(prices, scored, cfg, strategy)
        equity.to_csv(output / f'{strategy}_equity.csv', index=False)
        trades.to_csv(output / f'{strategy}_trades.csv', index=False)
        outputs[strategy] = summary
    scored[['date','symbol','setup','ai_score','rule_score','fold','atr_14',
            'close','relative_strength_20','relative_strength_60']].to_csv(
        output / 'out_of_sample_signals.csv', index=False)
    folds.to_csv(output / 'folds.csv', index=False)
    report = {'strategy': 'Nifty 500 market-filtered breakout/pullback with AI vs non-AI rankings',
              'config': asdict(cfg), 'price_policy': price_policy,
              'candidate_count': len(scored), 'fold_count': len(folds),
              'benchmark': 'Nifty 500 price index; not TRI, no exact trade-cost parity',
              'results': outputs, 'live_execution_enabled': False,
              'limitations': [
                  'Unadjusted data is for exploratory research only; corporate actions can distort rankings and stops.'
                  if price_policy == 'research-unadjusted' else
                  'Price-adjustment policy is user attested; independently verify actions and data provenance.',
                  'Constituent snapshot is not point-in-time; survivorship and delisting biases remain.',
                  'Stop execution is estimated from daily OHLC, not tick data; slippage may exceed assumptions.',
                  'Flat bps per side is a configurable approximation; real NSE fees, taxes and impact vary.',
                  'AI ranking scores are not probability or guaranteed return.',
                  'Optional sector map is static unless point-in-time membership is supplied.',
                  'Unobserved suspension during holdings fails closed rather than forward-filling.',
                  'End-of-sample holdings are liquidated at observed final close.',
              ]}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', required=True, help='Validated OHLCV CSV or directory with ohlcv_YYYY.csv.gz')
    parser.add_argument('--benchmark', required=True, help='CSV with date,nifty500_close')
    parser.add_argument('--output', default='artifacts/ai_swing')
    parser.add_argument('--sectors', help='Optional symbol,sector CSV (sector filter)')
    parser.add_argument('--price-policy', required=True, choices=['verified-adjusted','research-unadjusted'])
    for key in SwingResearchConfig.__dataclass_fields__:
        default = getattr(SwingResearchConfig(), key)
        parser.add_argument('--' + key.replace('_', '-'), type=type(default), default=default)
    args = parser.parse_args()
    opts = vars(args)
    cfg = SwingResearchConfig(**{key: opts[key] for key in SwingResearchConfig.__dataclass_fields__})
    report = run(opts['prices'], opts['benchmark'], opts['output'], cfg,
                 opts['price_policy'], opts['sectors'])
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()