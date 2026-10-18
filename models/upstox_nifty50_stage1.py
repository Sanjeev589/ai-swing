"""Frozen 2000–2016 model, untouched 2017–2019 Nifty50 holdout (paper research).

Stage 1 is deliberately not a hyperparameter search on the holdout. Never places orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from marketdata.shared_store import data_root, load_daily, load_universe

TRAIN_BEGIN = '2000-01-01'
TRAIN_END = '2016-12-31'
VALID_BEGIN = '2017-01-01'
VALID_END = '2019-12-31'


def _eligible_features(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    from features.technical_indicators import build_features, FEATURES
    frame = build_features(raw[['date','symbol','open','high','low','close','volume']], horizon=horizon)
    # A missing trading session can silently shorten a position's effective label.
    # A long suspension breaks warm-ups and forward labels; exclude nearby examples.
    gaps = frame.groupby('symbol').date.diff().dt.days.fillna(1).gt(14)
    frame['recent_gap'] = gaps.groupby(frame.symbol).transform(
        lambda s: s.rolling(55, min_periods=1).max()).astype(bool)
    frame['forward_gap'] = (frame.label_end_date - frame.date).dt.days.gt((horizon+1)*4 + 7)
    frame['extreme_jump'] = frame.groupby('symbol').close.pct_change(fill_method=None).abs().gt(.30)
    frame['recent_extreme_jump'] = frame.extreme_jump.groupby(frame.symbol).transform(
        lambda s: s.rolling(55, min_periods=1).max()).astype(bool)
    frame['eligible'] = (frame[FEATURES].notna().all(axis=1) & ~frame.recent_gap &
                         ~frame.forward_gap & ~frame.recent_extreme_jump &
                         frame.label_end_date.notna() & frame.target_return.notna())
    return frame


def _paper_cohorts(predictions: pd.DataFrame, *, rank_by: str, sessions: pd.DatetimeIndex,
                   top_k: int, horizon: int, side_cost_bps: float):
    """Non-overlapping next-open/next-open cohort returns; no intraday stop simulation."""
    side = side_cost_bps / 10000
    allocations = []
    start_at = sessions.searchsorted(pd.Timestamp(VALID_BEGIN))
    for i in range(start_at, len(sessions), horizon):
        if i + horizon + 1 >= len(sessions):
            break
        signal_date = sessions[i]
        expected_entry, expected_exit = sessions[i+1], sessions[i+horizon+1]
        batch = predictions[predictions.date.eq(signal_date) &
                            predictions.entry_date.eq(expected_entry) &
                            predictions.label_end_date.eq(expected_exit)]
        if batch.empty:
            continue
        selected = batch.sort_values([rank_by,'symbol'], ascending=[False,True]).head(top_k)
        for x in selected.itertuples(index=False):
            gross = float(x.target_return)
            if not np.isfinite(gross):
                continue
            allocations.append({'signal_date':signal_date, 'symbol':x.symbol,
                                'entry_date':x.entry_date,'exit_date':x.label_end_date,
                                'ranker':rank_by, 'score':float(getattr(x,rank_by)),
                                'gross_return':gross,
                                'net_return':(1+gross)*(1-side)/(1+side)-1})
    trades = pd.DataFrame(allocations)
    if trades.empty:
        raise ValueError('No completed holdout trades for '+rank_by)
    cohort = trades.groupby('signal_date',sort=True).net_return.mean()
    equity = (1+cohort).cumprod()
    drawdowns = equity / np.maximum.accumulate(np.r_[1.,equity.to_numpy()])[1:] - 1
    win = trades.net_return.gt(0)
    out = {'completed_trades':int(len(trades)), 'winning_trades':int(win.sum()),
           'losing_trades':int(trades.net_return.lt(0).sum()),
           'break_even_trades':int(trades.net_return.eq(0).sum()),
           'win_rate':float(win.mean()), 'average_trade_net_return':float(trades.net_return.mean()),
           'portfolio_cohort_compound_return':float(equity.iloc[-1]-1),
           'cohort_max_drawdown':float(drawdowns.min()),
           'cohorts':int(len(cohort)), 'cost_bps_per_side':side_cost_bps,
           'execution':'next-session open to next-session open, fixed holding duration',
           'not_included':'No intraday stops, market impact, actual NSE charges or execution spread'}
    return trades, out



def verify_download_audits(root: Path, snapshot: str) -> list[dict]:
    """Fail closed unless every requested stock/year was checked without HTTP errors."""
    audits=[]
    for start,end,years in ((TRAIN_BEGIN,TRAIN_END,17),(VALID_BEGIN,VALID_END,3)):
        filename = f'nifty50_{snapshot}_{start}_{end}_summary.json'
        path = root/'reports'/'upstox_v3'/filename
        if not path.exists():
            raise ValueError(f'Missing Upstox download audit: {path}; finish both downloads first')
        report=json.loads(path.read_text())
        expected=50*years
        if (report.get('universe_size') != 50 or
            report.get('requests_or_cache_checks') != expected or
            report.get('errors') != 0 or
            report.get('snapshot') != snapshot or
            report.get('start') != start or report.get('end') != end):
            raise ValueError(f'Incomplete/failed Upstox download audit: {path}')
        audits.append(report)
    return audits


def train_and_validate(*, root, snapshot: str, output: str = 'artifacts/upstox_nifty50_stage1',
                       horizon: int = 5, top_k: int = 5, estimators: int = 100,
                       cost_bps_per_side: float = 25., allow_unverified_corporate_actions: bool = False):
    """Fits on label-matured pre-2017 samples; 2017–2019 only scores and evaluates."""
    if min(horizon, top_k, estimators) < 1 or cost_bps_per_side < 0 or cost_bps_per_side >= 10000:
        raise ValueError('Invalid model, holding period, top-k or cost settings')
    if not allow_unverified_corporate_actions:
        raise ValueError('Upstox price adjustment is unverified; pass --allow-unverified-corporate-actions for exploratory research only')
    root = data_root(root)
    audits = verify_download_audits(root, snapshot)
    constituents = load_universe(root, snapshot)
    raw = load_daily(root, constituents, TRAIN_BEGIN, VALID_END)
    frame = _eligible_features(raw, horizon)
    feature_names = __import__('features.technical_indicators', fromlist=['FEATURES']).FEATURES
    train = frame[(frame.date <= pd.Timestamp(TRAIN_END)) &
                  (frame.label_end_date <= pd.Timestamp(TRAIN_END)) & frame.eligible].copy()
    holdout = frame[frame.date.between(VALID_BEGIN, VALID_END) & frame.eligible &
                    (frame.label_end_date <= pd.Timestamp(VALID_END))].copy()
    if train.empty or holdout.empty:
        raise ValueError('Insufficient 2000–2016 training or 2017–2019 validation rows')
    if train.date.nunique() < 252 or train.symbol.nunique() < 2 or holdout.symbol.nunique() < 2:
        raise ValueError('Need at least 252 training sessions and two eligible training/validation stocks')
    if train.label_end_date.max() >= holdout.date.min():
        raise ValueError('Train-to-holdout target leakage detected')
    from models.ranking import fit_ranker, rank_stocks
    ranker = fit_ranker(train, n_estimators=estimators)
    predictions = rank_stocks(ranker, holdout)
    predictions['rule_score'] = predictions.return_20.fillna(-999.)
    sessions = pd.DatetimeIndex(sorted(raw.loc[raw.date.between(VALID_BEGIN, VALID_END), 'date'].unique()))
    output = Path(output)
    if root == output.resolve() or root in output.resolve().parents:
        raise ValueError('Model reports must remain outside the shared raw market-data store')
    results, trades_all = {}, []
    for name in ('score','rule_score'):
        trades, metrics = _paper_cohorts(predictions, rank_by=name, sessions=sessions,
                                         top_k=top_k, horizon=horizon, side_cost_bps=cost_bps_per_side)
        results['lightgbm' if name == 'score' else 'momentum_20_baseline'] = metrics
        trades_all.append(trades)
    output.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(output/'ranker.txt'))
    predictions[['date','symbol','score','rule_score','rank','entry_date',
                 'label_end_date','target_return']].to_csv(output/'holdout_predictions.csv',index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(output/'holdout_trades.csv',index=False)
    coverage = raw.groupby('symbol').agg(first_session=('date','min'),last_session=('date','max'),
                                          observations=('date','size')).reset_index()
    coverage.to_csv(output/'stock_coverage.csv', index=False)
    summary = {'experiment':'stage1_fixed_nifty50_snapshot_2000_2019',
               'universe_snapshot':snapshot,'current_nifty50_not_historical_membership':True,
               'training_period':[TRAIN_BEGIN, TRAIN_END], 'held_out_period':[VALID_BEGIN,VALID_END],
               'training_label_end_latest':str(train.label_end_date.max().date()),
               'train_rows':int(len(train)), 'train_sessions':int(train.date.nunique()),
               'train_symbols':int(train.symbol.nunique()),'holdout_rows':int(len(holdout)),
               'holdout_sessions':int(holdout.date.nunique()),'holdout_symbols':int(holdout.symbol.nunique()),
               'horizon_sessions':horizon,'top_k':top_k, 'features':list(feature_names),
               'estimators':estimators,'results':results,'future_test_2020_onwards_untouched':True,
               'prices_corporate_actions':'UNVERIFIED - RESEARCH ONLY',
               'validated_download_audits':[{'start':a['start'],'end':a['end'],
                    'request_checks':a['requests_or_cache_checks'],'errors':a['errors'],
                    'no_data':a.get('no_data',0)} for a in audits],
               'data_license':'Verify broker permissions for reuse across projects',
               'limitations':['Current Nifty 50 contains survivorship bias in older years.',
                              'Not all present Nifty 50 stocks traded in 2000.',
                              'Split/bonus-adjusted price and volume history not independently verified.',
                              'Fixed-duration next-open simulated trades with approximate costs, not live fills.',
                              'Validation is a single untouched holdout; no parameter tuning on this period.',
                              'Incomplete provider years and historical symbols may affect coverage.',
                              'No live order execution.'],
               'live_execution_enabled':False,
               'python':platform.python_version(),
               'model_sha256':hashlib.sha256((output/'ranker.txt').read_bytes()).hexdigest()}
    (output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=str)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', help='Shared data folder OUTSIDE ai-swing')
    p.add_argument('--snapshot', required=True, help='Current Nifty50 snapshot date stored by downloader')
    p.add_argument('--output', default='artifacts/upstox_nifty50_stage1')
    p.add_argument('--horizon',type=int,default=5);p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--estimators',type=int,default=100);p.add_argument('--cost-bps-per-side',type=float,default=25.)
    p.add_argument('--allow-unverified-corporate-actions',action='store_true')
    args=p.parse_args()
    print(json.dumps(train_and_validate(root=args.data_root,snapshot=args.snapshot,output=args.output,
               horizon=args.horizon,top_k=args.top_k,estimators=args.estimators,
               cost_bps_per_side=args.cost_bps_per_side,
               allow_unverified_corporate_actions=args.allow_unverified_corporate_actions),indent=2))


if __name__=='__main__':
    main()