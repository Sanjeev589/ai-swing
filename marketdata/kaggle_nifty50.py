"""Audit extracted Rohan Rao Nifty-50 CSVs against Yahoo; store raw Kaggle independently.

Run from the ai-swing repository root::

    python -m marketdata.kaggle_nifty50 --snapshot YYYY-MM-DD

Input defaults to INDIA_MARKET_DATA_HOME/kaggle (extracted *files*, not zip).
NEVER modifies Yahoo adjusted partitions, canonical research_ohlcv or a trained model.
Kaggle OHLCV is historically unadjusted; cross-source price discrepancies are
expected around splits/bonus/dividends. Review/adjust before promoting any data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.shared_store import bar_path, data_root, load_universe, save_year, validate_bars

SOURCE = 'kaggle_raw'
# Kaggle's CSV for NSE symbol M&M is named MM.csv.
CSV_ALIASES = {'M&M': 'MM.csv'}
FIELDS = ('Date', 'Symbol', 'Series', 'Open', 'High', 'Low', 'Close', 'Volume')


def read_kaggle(path: Path, *, symbol: str, isin: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse a single stock's historic CSV using its *filename* as current identity.

    Kaggle's Symbol column may contain older NSE symbols for the same company.
    Validate the ISIN from stock_metadata.csv before reaching this function.
    """
    frame = pd.read_csv(path, low_memory=False)
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    if frame.empty:
        return pd.DataFrame(), []
    frame['Series'] = frame.Series.astype(str).str.strip().str.upper()
    frame = frame.loc[frame.Series.eq('EQ')].copy()
    if frame.empty:
        return pd.DataFrame(), []
    previous_symbols = sorted(frame.Symbol.dropna().astype(str).str.strip().str.upper().unique().tolist())
    if any(not x or x == 'NAN' for x in previous_symbols):
        raise ValueError(f'{path.name}: empty historic symbol')
    normalized = frame.rename(columns={'Date':'date','Open':'open','High':'high',
                                       'Low':'low','Close':'close','Volume':'volume'})
    normalized['symbol'] = symbol
    normalized['isin'] = isin
    return validate_bars(normalized), previous_symbols


def _metadata(directory: Path) -> dict[str, str]:
    path = directory / 'stock_metadata.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing Kaggle identity file: {path}')
    frame = pd.read_csv(path, dtype=str).fillna('')
    if not {'Symbol', 'ISIN Code'}.issubset(frame.columns) or frame.Symbol.duplicated().any():
        raise ValueError('Kaggle stock_metadata.csv must contain unique Symbol and ISIN Code')
    return dict(zip(frame.Symbol.str.strip().str.upper(),
                    frame['ISIN Code'].str.strip().str.upper()))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
    temp.replace(path)


def audit(*, root=None, snapshot: str, universe_name: str = 'nifty50',
          kaggle_dir=None, cache_raw: bool = True, price_tolerance_pct: float = 1.0) -> dict:
    """Compare per-symbol/year Kaggle raw files with cached Yahoo adjusted OHLCV.

    Changes only the kaggle_raw cache and audit reports. Existing Yahoo and
    research_ohlcv bytes remain untouched, even with --cache-raw.
    """
    if not np.isfinite(price_tolerance_pct) or price_tolerance_pct < 0:
        raise ValueError('Price tolerance must be nonnegative and finite')
    root = data_root(root)
    folder = Path(kaggle_dir) if kaggle_dir else root / 'kaggle'
    if not folder.is_dir():
        raise FileNotFoundError(f'Extract Kaggle archive files into {folder}')
    meta = _metadata(folder)
    universe = load_universe(root, snapshot, universe_name=universe_name)
    reports = root / 'reports' / SOURCE
    records, gaps, comparisons, failures = [], [], [], []
    members = set(universe.symbol)
    for item in universe.itertuples(index=False):
        source = folder / CSV_ALIASES.get(item.symbol, item.symbol + '.csv')
        record = {'symbol':item.symbol,'isin':item.isin,'kaggle_file':source.name,
                  'status':'','kaggle_first':'','kaggle_last':'','kaggle_rows':0,
                  'historical_symbols':'','kaggle_isin':meta.get(item.symbol,''),'error':''}
        if not source.is_file():
            record['status'] = 'not_in_kaggle'
            records.append(record)
            continue
        if not meta.get(item.symbol):
            record.update(status='missing_metadata',error='Kaggle stock_metadata.csv has no matching Symbol/ISIN')
            records.append(record)
            continue
        if meta[item.symbol] != item.isin:
            record.update(status='isin_mismatch',error='Kaggle historical ISIN differs from frozen Nifty universe; review corporate identity')
            records.append(record)
            continue
        try:
            bars, aliases = read_kaggle(source, symbol=item.symbol, isin=item.isin)
            if bars.empty:
                record['status'] = 'empty_file'
                records.append(record)
                continue
            record.update(status='checked',kaggle_first=bars.date.min().strftime('%Y-%m-%d'),
                          kaggle_last=bars.date.max().strftime('%Y-%m-%d'),
                          kaggle_rows=len(bars), historical_symbols=';'.join(aliases))
            for year, part in bars.groupby(bars.date.dt.year, sort=True):
                year = int(year)
                part = part.reset_index(drop=True)
                yahoo_path = bar_path(root,item.isin,year,'yahoo_finance')
                yahoo = None
                if yahoo_path.is_file():
                    yahoo = validate_bars(pd.read_csv(yahoo_path, dtype={'isin':str,'symbol':str}))
                    if (yahoo['isin'].iloc[0] != item.isin or yahoo.symbol.iloc[0] != item.symbol or
                        not yahoo.date.dt.year.eq(year).all()):
                        raise ValueError(f'Yahoo cache identity mismatch: {yahoo_path}')
                if cache_raw:
                    dest = bar_path(root,item.isin,year,SOURCE)
                    # Don't silently overwrite a previously audited Kaggle file.
                    save_year(root,part,isin=item.isin,year=year,provider=SOURCE)
                else:
                    dest = None
                kdays = part.date
                if yahoo is None:
                    missing = part
                    overlap = pd.DataFrame()
                else:
                    missing = part.loc[~kdays.isin(yahoo.date)]
                    overlap = part[['date','close','volume']].merge(
                        yahoo[['date','close','volume']],on='date',suffixes=('_kaggle','_yahoo'))
                n_mismatch = 0
                median_relative = None
                max_relative = None
                if not overlap.empty:
                    relative = 100.0 * (overlap.close_kaggle / overlap.close_yahoo - 1.0)
                    if not np.isfinite(relative).all():
                        raise ValueError(f'Nonfinite Yahoo/Kaggle ratio for {item.symbol} {year}')
                    n_mismatch = int(relative.abs().gt(price_tolerance_pct).sum())
                    median_relative = float(relative.median())
                    max_relative = float(relative.abs().max())
                comparisons.append({
                    'symbol':item.symbol,'isin':item.isin,'year':year,
                    'kaggle_rows':len(part),'yahoo_rows':0 if yahoo is None else len(yahoo),
                    'overlap_dates':len(overlap),'kaggle_only_dates':len(missing),
                    'median_close_diff_pct':median_relative,'max_absolute_close_diff_pct':max_relative,
                    'close_differences_over_tolerance':n_mismatch,
                    'raw_prices_unadjusted':True,
                    'eligible_for_automatic_adjusted_merge':False})
                for dt in missing.date:
                    gaps.append({'symbol':item.symbol,'isin':item.isin,'date':dt.strftime('%Y-%m-%d'),
                                 'year':year,'reason':'missing_yahoo_partition' if yahoo is None else 'missing_yahoo_session',
                                 'source':'kaggle_raw', 'requires_corporate_action_adjustment':True})
            records.append(record)
        except Exception as exc:
            record.update(status='error',error=f'{type(exc).__name__}: {str(exc)[:300]}')
            failures.append(record.copy())
            records.append(record)
    member_files = {CSV_ALIASES.get(symbol, symbol + '.csv') for symbol in members}
    outside = sorted(p.stem for p in folder.glob('*.csv')
                     if p.name not in member_files and p.name not in ('NIFTY50_all.csv','stock_metadata.csv'))
    _atomic_csv(reports/f'{universe_name}_{snapshot}_coverage.csv',pd.DataFrame(records))
    _atomic_csv(reports/f'{universe_name}_{snapshot}_comparisons.csv',pd.DataFrame(comparisons,columns=[
        'symbol','isin','year','kaggle_rows','yahoo_rows','overlap_dates','kaggle_only_dates',
        'median_close_diff_pct','max_absolute_close_diff_pct',
        'close_differences_over_tolerance','raw_prices_unadjusted','eligible_for_automatic_adjusted_merge']))
    _atomic_csv(reports/f'{universe_name}_{snapshot}_missing_yahoo_dates.csv',pd.DataFrame(gaps,columns=[
        'symbol','isin','date','year','reason','source','requires_corporate_action_adjustment']))
    _atomic_csv(reports/f'{universe_name}_{snapshot}_errors.csv',pd.DataFrame(failures,columns=list(records[0]) if records else []))
    result = {'provider':SOURCE,'universe_name':universe_name,'snapshot':snapshot,
              'kaggle_directory':str(folder),'metadata_symbols':len(meta),
              'universe_symbols':len(universe),'checked':sum(r['status']=='checked' for r in records),
              'unavailable_in_kaggle':sum(r['status']=='not_in_kaggle' for r in records),
              'isin_mismatches':sum(r['status']=='isin_mismatch' for r in records),
              'empty_files':sum(r['status']=='empty_file' for r in records),
              'errors':len(failures),'kaggle_only_dates':len(gaps),
              'yahoo_comparison_years':len(comparisons),
              'kaggle_names_not_in_current_universe':outside,
              'raw_partitions_cached':cache_raw,
              'yahoo_adjusted_partitions_modified':False,
              'research_ohlcv_partitions_modified':False,
              'warning':'Kaggle data is raw, NOT verified adjusted. Never combine it directly with Yahoo adjusted OHLCV.',
              'report_directory':str(reports),
              'audited_at_utc':datetime.now(timezone.utc).isoformat()}
    _atomic_json(reports/f'{universe_name}_{snapshot}_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',help='External shared-data root')
    p.add_argument('--snapshot',required=True,help='Existing frozen universe snapshot date')
    p.add_argument('--universe-name',default='nifty50')
    p.add_argument('--kaggle-dir',help='Extract Kaggle archive here; defaults to INDIA_MARKET_DATA_HOME/kaggle')
    p.add_argument('--no-cache-raw',action='store_true',help='Audit without writing raw Kaggle partitions')
    p.add_argument('--price-tolerance-pct',type=float,default=1.0)
    args=p.parse_args()
    result=audit(root=args.data_root,snapshot=args.snapshot,universe_name=args.universe_name,
                 kaggle_dir=args.kaggle_dir,cache_raw=not args.no_cache_raw,
                 price_tolerance_pct=args.price_tolerance_pct)
    print(json.dumps(result,indent=2))
    if result['errors']:
        raise SystemExit(2)

if __name__=='__main__':
    main()
