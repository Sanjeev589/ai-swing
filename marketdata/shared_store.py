"""Reusable, project-independent on-disk daily OHLCV store.

Physical data lives outside any strategy repository. All writes are atomic and keyed by
provider + ISIN + year; a dated universe snapshot records the *current* membership.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

COLUMNS = ('date', 'symbol', 'isin', 'instrument_key', 'open', 'high', 'low', 'close', 'volume', 'open_interest')


def data_root(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get('INDIA_MARKET_DATA_HOME')
    if not raw:
        raise ValueError('Provide --data-root or INDIA_MARKET_DATA_HOME; data must be outside the ai-swing repository')
    root = Path(raw).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if root == repo_root or repo_root in root.parents:
        raise ValueError('Shared market data must be stored OUTSIDE the ai-swing repository')
    return root


def bar_path(root: Path, isin: str, year: int, provider: str = 'upstox_v3') -> Path:
    if not str(isin).startswith('IN') or not str(isin).isalnum() or not 1990 <= year <= 2100:
        raise ValueError('Invalid ISIN or year')
    return root / 'candles' / provider / 'NSE_EQ' / '1d' / isin / f'{year}.csv.gz'


def validate_bars(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing daily-data columns: {sorted(missing)}')
    out = df.loc[:, COLUMNS].copy()
    out['date'] = pd.to_datetime(out.date, errors='raise')
    if out.date.isna().any() or out.date.dt.tz is not None or not out.date.eq(out.date.dt.normalize()).all():
        raise ValueError('Expected valid, timezone-naive NSE session dates')
    if out.date.duplicated().any():
        raise ValueError('Duplicate session in instrument partition')
    if symbol is not None and (out.symbol.ne(symbol)).any():
        raise ValueError('Symbol mismatch in instrument partition')
    for col in ('open', 'high', 'low', 'close', 'volume', 'open_interest'):
        out[col] = pd.to_numeric(out[col], errors='raise')
    if not np.isfinite(out[['open', 'high', 'low', 'close', 'volume', 'open_interest']].to_numpy()).all():
        raise ValueError('Non-finite prices or volume')
    if (out[['open', 'high', 'low', 'close']] <= 0).any().any() or (out[['volume', 'open_interest']] < 0).any().any():
        raise ValueError('Invalid OHLCV magnitude')
    if (out.high < out[['open','low','close']].max(axis=1)).any() or (out.low > out[['open','high','close']].min(axis=1)).any():
        raise ValueError('Inconsistent daily OHLC')
    return out.sort_values('date').reset_index(drop=True)


def _atomic_csv_gz(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent, prefix='.pending-', suffix='.csv.gz', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # Set compression timestamp to 0 for reproducible bytes and avoid exposing temp filename.
        with tmp_path.open('wb') as handle:
            import gzip
            with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as zipped:
                zipped.write(df.to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8'))
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_year(root: Path, df: pd.DataFrame, *, isin: str, year: int) -> Path:
    checked = validate_bars(df)
    if checked.empty or checked['isin'].nunique() != 1 or checked['isin'].iloc[0] != isin or not checked.date.dt.year.eq(year).all():
        raise ValueError('A yearly partition must contain only one ISIN and year')
    path = bar_path(root, isin, year)
    _atomic_csv_gz(checked, path)
    return path


def load_universe(root: Path, snapshot: str) -> pd.DataFrame:
    path = root / 'universes' / 'nifty50' / f'{snapshot}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing immutable Nifty 50 snapshot: {path}')
    df = pd.read_csv(path, dtype={'isin': str, 'instrument_key': str, 'symbol': str})
    if len(df) != 50 or df.symbol.duplicated().any() or df.instrument_key.duplicated().any():
        raise ValueError('Expected 50 unique snapshot stocks')
    return df


def load_daily(root: Path, universe: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError('Invalid dates')
    fragments = []
    for row in universe.itertuples(index=False):
        for year in range(first.year, last.year + 1):
            path = bar_path(root, row.isin, year)
            if path.exists():
                bars = validate_bars(pd.read_csv(path))
                if not bars.empty and (bars['isin'].ne(row.isin).any() or bars.instrument_key.ne(row.instrument_key).any()):
                    raise ValueError(f'Inconsistent ISIN/instrument mapping in {path}')
                fragments.append(bars)
    if not fragments:
        raise ValueError('No downloaded daily bars in the requested period')
    combined = pd.concat(fragments, ignore_index=True)
    if combined.duplicated(['date', 'symbol']).any():
        raise ValueError('Duplicate symbol/date across shared-data partitions')
    selected = combined[combined.date.between(first, last)].copy()
    if selected.empty:
        raise ValueError('No bars in the requested period')
    return selected.sort_values(['symbol', 'date']).reset_index(drop=True)