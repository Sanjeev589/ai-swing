"""Project-independent, locally reusable daily NSE OHLCV store.

Data lives outside the ai-swing repository. No online access is needed to read it.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import tempfile
import re

import numpy as np
import pandas as pd

COLUMNS = ('date', 'symbol', 'isin', 'open', 'high', 'low', 'close', 'volume')
PROVIDERS = {'yahoo_finance', 'nse_raw', 'kaggle_raw', 'research_ohlcv'}


def data_root(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get('INDIA_MARKET_DATA_HOME')
    if not raw:
        raise ValueError('Supply --data-root or INDIA_MARKET_DATA_HOME (outside ai-swing)')
    root = Path(raw).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError('Shared market data must be outside the ai-swing repository')
    return root


def validate_universe_name(name: str) -> str:
    """Safe directory key; supports nifty50, nifty500 and future custom baskets."""
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,48}', name):
        raise ValueError('Universe name must use lowercase letters, digits, _ or -')
    return name


def bar_path(root: Path, isin: str, year: int, provider: str = 'yahoo_finance') -> Path:
    approved = provider in PROVIDERS or (isinstance(provider, str) and provider.startswith('ext_')
                and 5 <= len(provider) <= 48 and provider[4:].replace('_','').replace('-','').isalnum()
                and provider == provider.lower())
    if not approved or not (len(isin) == 12 and isin.startswith('IN') and isin.isalnum()):
        raise ValueError('Invalid provider or ISIN')
    if not (1990 <= year <= 2100):
        raise ValueError('Invalid year')
    return Path(root) / 'candles' / provider / 'NSE_EQ' / '1d' / isin / f'{year}.csv.gz'


def validate_bars(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing daily OHLCV columns: {sorted(missing)}')
    out = df.loc[:, COLUMNS].copy()
    out['date'] = pd.to_datetime(out.date, errors='raise')
    if (out.date.isna().any() or out.date.dt.tz is not None or
            not out.date.eq(out.date.dt.normalize()).all()):
        raise ValueError('Expected timezone-naive daily NSE session dates')
    if out.duplicated(['date','symbol']).any():
        raise ValueError('Duplicate symbol/date')
    if out.empty:
        raise ValueError('No daily bars')
    if out.symbol.isna().any() or out['isin'].isna().any():
        raise ValueError('Missing instrument identity')
    if out.symbol.nunique() != 1 or out['isin'].nunique() != 1:
        raise ValueError('Mixed symbols or ISINs in one partition')
    if symbol and out.symbol.iloc[0] != symbol:
        raise ValueError('Symbol mismatch')
    if len(out['isin'].iloc[0]) != 12 or not out['isin'].iloc[0].startswith('IN'):
        raise ValueError('Invalid ISIN')
    for name in ('open','high','low','close','volume'):
        out[name] = pd.to_numeric(out[name], errors='raise')
    if not np.isfinite(out[list(COLUMNS)[3:]].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite OHLCV')
    if out[['open','high','low','close']].le(0).any().any() or out.volume.lt(0).any():
        raise ValueError('Invalid OHLCV magnitude')
    if ((out.high < out[['open','low','close']].max(axis=1)).any() or
            (out.low > out[['open','high','close']].min(axis=1)).any()):
        raise ValueError('Inconsistent daily OHLC')
    return out.sort_values('date').reset_index(drop=True)


def save_year(root: Path, frame: pd.DataFrame, *, isin: str, year: int,
              provider: str = 'yahoo_finance', replace: bool = False) -> Path:
    checked = validate_bars(frame)
    if checked['isin'].iloc[0] != isin or not checked.date.dt.year.eq(year).all():
        raise ValueError('Partition does not match ISIN/year')
    dest = bar_path(root, isin, year, provider)
    if dest.exists() and not replace:
        cached = validate_bars(pd.read_csv(dest, dtype={'isin': str, 'symbol': str}))
        try:
            pd.testing.assert_frame_equal(cached, checked, check_dtype=False, rtol=1e-12, atol=1e-12)
        except AssertionError as exc:
            raise FileExistsError(f'Partition differs (use explicit refresh): {dest}') from exc
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=dest.parent, prefix='.pending-',
                                     suffix='.gz', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with temporary.open('wb') as handle:
            with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0) as zipped:
                zipped.write(checked.to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8'))
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def load_universe(root: Path, snapshot: str, universe_name: str = 'nifty50') -> pd.DataFrame:
    universe_name = validate_universe_name(universe_name)
    path = Path(root) / 'universes' / universe_name / f'{snapshot}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen {universe_name} snapshot: {path}')
    frame = pd.read_csv(path, dtype=str).fillna('')
    required = {'symbol','isin'}
    if not required.issubset(frame.columns) or not (set(frame.columns) & {'yahoo_ticker','instrument_key'}) or frame.empty or (universe_name == 'nifty50' and len(frame) != 50):
        raise ValueError('Snapshot requires symbols, ISINs and provider instrument identity; nifty50 needs 50 stocks')
    if any(frame[k].eq('').any() or frame[k].duplicated().any() for k in required):
        raise ValueError('Duplicate/missing universe identity')
    if not frame['isin'].map(lambda x: len(x)==12 and x.startswith('IN') and x.isalnum()).all():
        raise ValueError('Invalid snapshot ISINs')
    return frame


def load_daily(root: Path, universe: pd.DataFrame, start: str, end: str,
               provider: str = 'yahoo_finance') -> pd.DataFrame:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError('Invalid date range')
    fragments = []
    for item in universe.itertuples(index=False):
        for year in range(first.year, last.year+1):
            path = bar_path(root, item.isin, year, provider)
            if not path.is_file():
                continue
            bars = validate_bars(pd.read_csv(path, dtype={'symbol':str,'isin':str}))
            if bars['isin'].iloc[0] != item.isin or bars.symbol.iloc[0] != item.symbol or not bars.date.dt.year.eq(year).all():
                raise ValueError(f'Identity/year mismatch in {path}')
            fragments.append(bars)
    if not fragments:
        raise ValueError('No OHLCV files found; run the downloader first')
    combined = pd.concat(fragments, ignore_index=True)
    if combined.duplicated(['date','symbol']).any():
        raise ValueError('Duplicate stock/session across files')
    selected = combined[combined.date.between(first,last)].copy()
    if selected.empty:
        raise ValueError('No daily bars within selected years')
    return selected.sort_values(['symbol','date']).reset_index(drop=True)