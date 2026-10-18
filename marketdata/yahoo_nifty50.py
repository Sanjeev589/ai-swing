"""No-login Yahoo daily OHLCV downloader for today's Nifty 50 (research only).

Uses public Nifty Indices CSV for a DATED, fixed-universe snapshot. No broker login,
no order placement. Resumable by ISIN/year; explicit report distinguishes empty,
failed and cached years. Requires an external shared-data directory.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import io
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from marketdata.shared_store import bar_path, data_root, save_year, validate_bars, validate_universe_name

NIFTY50_CSV = 'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv'
NIFTY500_CSV = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
UNIVERSE_URLS = {'nifty50': NIFTY50_CSV, 'nifty500': NIFTY500_CSV}
PROVIDER = 'yahoo_finance'


def parse_constituents(csv_text: str, *, overrides: dict[str,str] | None = None,
                       expected_count: int | None = 50) -> pd.DataFrame:
    """Require an official 50-stock CSV with ISIN; don't use fabricated ticker lists."""
    df = pd.read_csv(io.StringIO(csv_text), dtype=str).fillna('')
    keys = {str(k).strip().casefold().replace(' ', ''): k for k in df.columns}
    sym_col = keys.get('symbol')
    isin_col = keys.get('isincode') or keys.get('isin')
    if sym_col is None or isin_col is None:
        raise ValueError('Nifty indices CSV must include Symbol and ISIN Code')
    if not len(df) or (expected_count is not None and len(df) != expected_count):
        raise ValueError(f'Universe count mismatch: expected {expected_count}, got {len(df)}')
    result = pd.DataFrame({'symbol':df[sym_col].str.strip().str.upper(),
                           'isin':df[isin_col].str.strip().str.upper()})
    if result.symbol.eq('').any() or result.symbol.duplicated().any():
        raise ValueError('Missing/duplicate Nifty 50 symbols')
    if result['isin'].duplicated().any() or not result['isin'].map(
            lambda v: len(v)==12 and v.startswith('IN') and v.isalnum()).all():
        raise ValueError('Missing/duplicate/invalid constituent ISIN')
    overrides = overrides or {}
    if set(overrides) - set(result.symbol):
        raise ValueError('Unknown symbol in ticker overrides')
    result['yahoo_ticker'] = result.symbol.map(lambda s: overrides.get(s, s+'.NS'))
    if (result.yahoo_ticker.eq('').any() or result.yahoo_ticker.duplicated().any() or
            not result.yahoo_ticker.str.endswith('.NS').all()):
        raise ValueError('Invalid or duplicate Yahoo NSE tickers')
    return result.sort_values('symbol').reset_index(drop=True)


def _download_constituents(session, url=NIFTY50_CSV, retries=3, pause=1.0) -> str:
    for attempt in range(retries):
        try:
            response=session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(10,40))
            if response.status_code in (429,500,502,503,504):
                raise requests.HTTPError(f'Nifty Indices HTTP {response.status_code}')
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError('Empty official constituent CSV')
            return response.text
        except (requests.RequestException, ValueError):
            if attempt+1 == retries: raise
            time.sleep(pause*(2**attempt))
    raise AssertionError('unreachable')


def freeze_snapshot(root: Path, *, asof: str, constituents: pd.DataFrame,
                    source: str = NIFTY50_CSV, universe_name: str = 'nifty50') -> Path:
    try:
        date.fromisoformat(asof)
    except (ValueError, TypeError) as exc:
        raise ValueError('--asof must be YYYY-MM-DD') from exc
    universe_name=validate_universe_name(universe_name)
    location=Path(root)/'universes'/universe_name
    location.mkdir(parents=True,exist_ok=True)
    dest=location/f'{asof}.csv'
    csv=constituents.to_csv(index=False)
    if dest.exists():
        if dest.read_text(encoding='utf-8') != csv:
            raise ValueError('Snapshot changed on the same as-of date: refuse overwrite')
    else:
        dest.write_text(csv,encoding='utf-8')
    meta=location/f'{asof}.json'
    if not meta.exists():
        meta.write_text(json.dumps({'asof':asof,'source':source,
            'snapshot_sha256':hashlib.sha256(csv.encode()).hexdigest(),
            'universe_type':f'current {universe_name} only (not historical membership)',
            'price_source':'Yahoo via yfinance; no official adjusted OHLCV guarantee'},indent=2)+'\n')
    return dest


def normalize_yahoo(raw: pd.DataFrame, *, symbol: str, isin: str, year: int) -> pd.DataFrame | None:
    if raw is None or raw.empty: return None
    df=raw.copy()
    if isinstance(df.columns,pd.MultiIndex):
        # For one ticker, flatten deterministic OHLCV level.
        fields={'Open','High','Low','Close','Volume'}
        levels=[i for i in range(df.columns.nlevels) if fields.issubset(set(df.columns.get_level_values(i)))]
        if len(levels)!=1: raise ValueError(f'Ambiguous Yahoo multiindex columns for {symbol}')
        df.columns=df.columns.get_level_values(levels[0])
    df.columns=[str(k).strip().lower() for k in df.columns]
    if not {'open','high','low','close','volume'}.issubset(df.columns):
        raise ValueError(f'Yahoo download missing OHLCV for {symbol}')
    days=pd.DatetimeIndex(pd.to_datetime(df.index,errors='raise'))
    if days.tz is not None:
        days=days.tz_convert('Asia/Kolkata').tz_localize(None)
    df['date']=days.normalize()
    df['symbol']=symbol
    df['isin']=isin
    bars=validate_bars(df)
    if not bars.date.dt.year.eq(year).all():
        raise ValueError(f'Yahoo returned bars outside requested year {year}')
    return bars


def _download_ticker(ticker, start, end, downloader, *, attempts=3, pause=2.0):
    for retry in range(attempts):
        try:
            raw=downloader(ticker,start=start,end=end,interval='1d',auto_adjust=True,
                           actions=False,progress=False,threads=False,
                           multi_level_index=False,timeout=30)
            if raw is None:
                raise ValueError('Yahoo downloader returned None')
            return raw
        except Exception:  # yfinance raises a variety of provider/network exceptions
            if retry == attempts-1: raise
            time.sleep(pause*2**retry)
    raise AssertionError('unreachable')


def _atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
    temp.replace(path)


def collect(*, root=None, start='2000-01-01', end='2016-12-31',
            asof=None, universe_csv=None, overrides_csv=None, downloader=None,
            session=None, pause=0.8, refresh=False, universe_name='nifty50'):
    """Download complete calendar years, inclusive end date; never mask failures as no-data."""
    begin, finish = date.fromisoformat(start), date.fromisoformat(end)
    if begin > finish or begin.month!=1 or begin.day!=1 or finish.month!=12 or finish.day!=31:
        raise ValueError('Supply full calendar-year boundaries, YYYY-01-01 through YYYY-12-31')
    if finish >= datetime.now(ZoneInfo('Asia/Kolkata')).date():
        raise ValueError('Only completed years may use the full-year downloader')
    if pause < 0: raise ValueError('pause must be nonnegative')
    root=data_root(root)
    universe_name=validate_universe_name(universe_name)
    if not universe_csv and universe_name not in UNIVERSE_URLS:
        raise ValueError('Custom universe requires --universe-csv containing Symbol and ISIN Code')
    asof=asof or datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()
    session=session or requests.Session()
    if universe_csv:
        csv_text=Path(universe_csv).read_text(encoding='utf-8-sig')
        origin='local supplied NSE constituent snapshot'
    else:
        origin=UNIVERSE_URLS[universe_name]
        csv_text=_download_constituents(session, url=origin)
    overrides={}
    if overrides_csv:
        mapped=pd.read_csv(overrides_csv,dtype=str).fillna('')
        if not {'symbol','yahoo_ticker'}.issubset(mapped.columns) or mapped.symbol.duplicated().any():
            raise ValueError('Override CSV needs unique symbol,yahoo_ticker columns')
        overrides=dict(zip(mapped.symbol,mapped.yahoo_ticker))
    universe=parse_constituents(csv_text,overrides=overrides,
                                expected_count=50 if universe_name=='nifty50' else None)
    freeze_snapshot(root,asof=asof,constituents=universe,source=origin,universe_name=universe_name)
    if downloader is None:
        import yfinance as yf
        downloader=yf.download
    status=[]
    for item in universe.itertuples(index=False):
        for year in range(begin.year,finish.year+1):
            dest=bar_path(root,item.isin,year,PROVIDER)
            outcome={'symbol':item.symbol,'isin':item.isin,'ticker':item.yahoo_ticker,
                     'year':year,'status':None,'rows':0,'error':''}
            try:
                if dest.is_file() and not refresh:
                    bars=validate_bars(pd.read_csv(dest,dtype={'isin':str}))
                    if bars.symbol.iloc[0]!=item.symbol or bars['isin'].iloc[0]!=item.isin or not bars.date.dt.year.eq(year).all():
                        raise ValueError('Existing cache identity/year mismatch')
                    outcome.update(status='cached',rows=len(bars))
                else:
                    start_date=f'{year}-01-01';end_date=f'{year+1}-01-01'
                    raw=_download_ticker(item.yahoo_ticker,start_date,end_date,downloader,pause=pause)
                    bars=normalize_yahoo(raw,symbol=item.symbol,isin=item.isin,year=year)
                    if bars is None:
                        # Empty can be pre-listing OR provider blocking; keep it distinct and auditable.
                        outcome['status']='no_data_unverified'
                    else:
                        save_year(root,bars,isin=item.isin,year=year,provider=PROVIDER,replace=refresh)
                        outcome.update(status='downloaded',rows=len(bars))
                    if pause: time.sleep(pause)
            except Exception as exc:
                outcome.update(status='error',error=f'{type(exc).__name__}: {str(exc)[:220]}')
            status.append(outcome)
    folder=root/'reports'/PROVIDER
    stem=f'{universe_name}_{asof}_{start}_{end}'
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(status).to_csv(folder/f'{stem}_status.csv',index=False)
    counts=pd.Series([r['status'] for r in status]).value_counts().to_dict()
    report={'provider':PROVIDER,'universe_name':universe_name,'start':start,'end':end,'snapshot':asof,
            'universe_size':len(universe),'requests_or_cache_checks':len(status),
            'downloaded':int(counts.get('downloaded',0)),
            'cached':int(counts.get('cached',0)),
            'no_data_unverified':int(counts.get('no_data_unverified',0)),
            'errors':int(counts.get('error',0)),
            'source':'Yahoo Finance via yfinance; auto_adjust=True',
            'adjustment_policy':'provider-adjusted OHLC; split/bonus/dividend effects require independent review',
            'historical_membership':f'current {universe_name} snapshot, NOT historic constituents',
            'status_csv':str(folder/f'{stem}_status.csv'),
            'live_trading_enabled':False}
    _atomic_json(folder/f'{stem}_summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',help='External market data root, overrides INDIA_MARKET_DATA_HOME')
    p.add_argument('--from-date',default='2000-01-01'); p.add_argument('--to-date',default='2016-12-31')
    p.add_argument('--asof',help='Dated current universe snapshot, default today IST')
    p.add_argument('--universe-name',default='nifty50',
                   help='nifty50 (default), nifty500, or a custom basket name with --universe-csv')
    p.add_argument('--universe-csv',help='Offline copy of NSE Nifty50 member CSV')
    p.add_argument('--overrides-csv',help='Optional symbol,yahoo_ticker mapping CSV')
    p.add_argument('--pause-seconds',type=float,default=.8)
    p.add_argument('--refresh',action='store_true',help='Explicitly redownload and replace existing full-year partitions')
    args=p.parse_args()
    report=collect(root=args.data_root,start=args.from_date,end=args.to_date,
                   asof=args.asof,universe_csv=args.universe_csv,
                   overrides_csv=args.overrides_csv,pause=args.pause_seconds,refresh=args.refresh,
                   universe_name=args.universe_name)
    print(json.dumps(report,indent=2))
    if report['errors']: raise SystemExit(2)

if __name__=='__main__':main()
