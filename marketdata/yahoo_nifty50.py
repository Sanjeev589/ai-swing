"""No-login Yahoo daily OHLCV downloader for today's Nifty 50 (research only).

Uses public Nifty Indices CSV for a DATED, fixed-universe snapshot. No broker login,
no order placement. Resumable by ISIN/year; explicit report distinguishes empty,
failed and cached years. Requires an external shared-data directory.
"""
from __future__ import annotations

import argparse
import csv
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

# Append-only: earlier failures remain visible after a successful retry.
ERROR_COLUMNS = (
    'logged_at_utc', 'universe_name', 'snapshot', 'company_name', 'symbol',
    'isin', 'ticker', 'year', 'from_date', 'to_date', 'status', 'error_type',
    'error_message', 'ignore',
)


def _issue_policy(path: Path, *, universe_name: str) -> set[tuple[str, int]]:
    """Migrate existing log and honor the LAST human-edited ignore value per ISIN/year.

    Ignore applies to this Yahoo collector only; independent providers may still
    fill the omitted stock-years. Existing success cache wins over ignore.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(ERROR_COLUMNS[:-1]).issubset(reader.fieldnames):
            raise ValueError(f'Invalid existing error CSV headers: {path}')
        names = list(reader.fieldnames)
        rows = list(reader)
    # Old error logs predate the ignore column. Preserve all earlier rows
    # and any user-added columns; add ignore=false once, atomically.
    if 'ignore' not in names:
        names.append('ignore')
        for row in rows:
            row['ignore'] = 'false'
        tmp = path.with_suffix('.migration.tmp')
        with tmp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    policy = {}
    for row in rows:
        raw = (row.get('ignore') or 'false').strip().lower()
        if raw not in ('true', 'false'):
            raise ValueError(f'ignore must be true/false in {path} for {row.get("symbol")} {row.get("year")}')
        if row.get('universe_name') != universe_name:
            continue
        try:
            key = (row['isin'].strip().upper(), int(row['year']))
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Invalid ISIN/year in {path}') from exc
        policy[key] = (raw == 'true')
    return {key for key, ignored in policy.items() if ignored}


def _append_download_issue(path: Path, record: dict) -> None:
    """Append failed/empty requests, keeping human-added ignore flags intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open('r', newline='', encoding='utf-8-sig') as handle:
            fields = list(csv.DictReader(handle).fieldnames or [])
        if 'ignore' not in fields:
            raise ValueError('Existing error log needs ignore-column migration first')
    else:
        fields = list(ERROR_COLUMNS)
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({**record, 'ignore': 'false'})
        handle.flush()


def _company_names(csv_text: str) -> dict[str, str]:
    """Use official company names for diagnostics without changing frozen snapshots."""
    frame = pd.read_csv(io.StringIO(csv_text), dtype=str).fillna('')
    keys = {str(k).strip().casefold().replace(' ', ''): k for k in frame.columns}
    symbol_col = keys.get('symbol')
    name_col = keys.get('companyname') or keys.get('name')
    if symbol_col is None or name_col is None:
        return {}
    return dict(zip(frame[symbol_col].str.strip().str.upper(), frame[name_col].str.strip()))



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
    csv=constituents.to_csv(index=False,lineterminator='\n')
    if dest.exists():
        # Cross-platform logical comparison: Windows CRLF must not create a false
        # same-day snapshot conflict on a subsequent download invocation.
        observed=pd.read_csv(dest,dtype=str).fillna('')
        if observed.to_dict('records') != constituents.to_dict('records'):
            raise ValueError('Snapshot changed on the same as-of date: refuse overwrite')
    else:
        dest.write_bytes(csv.encode('utf-8'))
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
            session=None, pause=0.8, refresh=False, universe_name='nifty50', only_symbols=None):
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
    company_names = _company_names(csv_text)
    # Optional diagnostic subset never replaces a full-universe audit report.
    chosen = None
    if only_symbols:
        chosen = {str(x).strip().upper() for x in only_symbols}
        if not chosen or chosen - set(universe.symbol):
            raise ValueError(f'Unknown --only-symbols: {sorted(chosen - set(universe.symbol))}')
        selected = universe[universe.symbol.isin(chosen)]
    else:
        selected = universe
    folder = root/'reports'/PROVIDER
    folder.mkdir(parents=True, exist_ok=True)
    error_log = folder/f'{universe_name}_download_errors.csv'
    ignored_stock_years = _issue_policy(error_log, universe_name=universe_name)
    if downloader is None:
        import yfinance as yf
        downloader=yf.download
    status=[]
    for item in selected.itertuples(index=False):
        for year in range(begin.year,finish.year+1):
            dest=bar_path(root,item.isin,year,PROVIDER)
            year_start=f'{year}-01-01'
            year_end=f'{year}-12-31'
            outcome={'symbol':item.symbol,'company_name':company_names.get(item.symbol,item.symbol),
                     'isin':item.isin,'ticker':item.yahoo_ticker,
                     'year':year,'from_date':year_start,'to_date':year_end,
                     'status':None,'rows':0,'error':''}
            error_type=''
            try:
                if dest.is_file() and not refresh:
                    # A valid completed stock/year is ALWAYS skipped on subsequent runs.
                    bars=validate_bars(pd.read_csv(dest,dtype={'isin':str}))
                    if bars.symbol.iloc[0]!=item.symbol or bars['isin'].iloc[0]!=item.isin or not bars.date.dt.year.eq(year).all():
                        raise ValueError('Existing cache identity/year mismatch')
                    outcome.update(status='cached',rows=len(bars))
                elif (item.isin, year) in ignored_stock_years:
                    outcome.update(status='ignored', error='Manually marked ignore=true in Yahoo error CSV')
                else:
                    raw=_download_ticker(item.yahoo_ticker,year_start,f'{year+1}-01-01',downloader,pause=pause)
                    bars=normalize_yahoo(raw,symbol=item.symbol,isin=item.isin,year=year)
                    if bars is None:
                        # Empty data are NOT assumed pre-listing; retry on next run.
                        outcome.update(status='no_data_unverified',
                                       error='Yahoo returned no bars (pre-listing, symbol change, throttling or outage)')
                        error_type='NoDataUnverified'
                    else:
                        save_year(root,bars,isin=item.isin,year=year,provider=PROVIDER,replace=refresh)
                        outcome.update(status='downloaded',rows=len(bars))
                    if pause: time.sleep(pause)
            except Exception as exc:
                error_type=type(exc).__name__
                outcome.update(status='error',error=f'{error_type}: {str(exc)[:220]}')
            if outcome['status'] in ('error','no_data_unverified'):
                _append_download_issue(error_log,{
                    'logged_at_utc':datetime.now(ZoneInfo('UTC')).isoformat(),
                    'universe_name':universe_name,'snapshot':asof,
                    **outcome,'error_type':error_type,'error_message':outcome['error']})
                print(f"{outcome['status'].upper()} {item.symbol} {year_start}..{year_end}: {outcome['error']}",flush=True)
            status.append(outcome)
    stem=f'{universe_name}_{asof}_{start}_{end}'
    if chosen is not None:
        stem += '_subset_' + hashlib.sha256(','.join(sorted(chosen)).encode()).hexdigest()[:10]
    pd.DataFrame(status).to_csv(folder/f'{stem}_status.csv',index=False)
    counts=pd.Series([r['status'] for r in status]).value_counts().to_dict()
    report={'provider':PROVIDER,'universe_name':universe_name,'start':start,'end':end,'snapshot':asof,
            'universe_size':len(universe),'selected_symbols':len(selected),
            'partial_universe':chosen is not None, 'requests_or_cache_checks':len(status),
            'downloaded':int(counts.get('downloaded',0)),
            'cached':int(counts.get('cached',0)),
            'ignored':int(counts.get('ignored',0)),
            'no_data_unverified':int(counts.get('no_data_unverified',0)),
            'errors':int(counts.get('error',0)),
            'source':'Yahoo Finance via yfinance; auto_adjust=True',
            'adjustment_policy':'provider-adjusted OHLC; split/bonus/dividend effects require independent review',
            'historical_membership':f'current {universe_name} snapshot, NOT historic constituents',
            'status_csv':str(folder/f'{stem}_status.csv'),
            'persistent_error_csv':str(error_log),
            'cached_stock_years_not_redownloaded':int(counts.get('cached',0)),
            'ignored_stock_years_not_redownloaded':int(counts.get('ignored',0)),
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
    p.add_argument('--only-symbols',nargs='+',help='Diagnose a subset (e.g. RELIANCE TCS) while keeping full Nifty50 snapshot')
    p.add_argument('--refresh',action='store_true',help='Explicitly redownload and replace existing full-year partitions')
    args=p.parse_args()
    report=collect(root=args.data_root,start=args.from_date,end=args.to_date,
                   asof=args.asof,universe_csv=args.universe_csv,
                   overrides_csv=args.overrides_csv,pause=args.pause_seconds,refresh=args.refresh,
                   universe_name=args.universe_name,only_symbols=args.only_symbols)
    print(json.dumps(report,indent=2))
    if report['errors']: raise SystemExit(2)

if __name__=='__main__':main()