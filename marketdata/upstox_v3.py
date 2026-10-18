"""Upstox V3 daily NSE collector for an immutable *current* Nifty 50 snapshot.

Never places orders. Never stores tokens. Uses explicit external shared-data root.
Historical API max for daily data is one decade; yearly slices simplify resume/audit.
"""
from __future__ import annotations

import argparse
from datetime import date
from io import BytesIO, StringIO
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

import pandas as pd
import requests
from marketdata.shared_store import COLUMNS, bar_path, data_root, save_year, validate_bars

NIFTY50_URL = 'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv'
UPSTOX_NSE_INSTRUMENTS_URL = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
API = 'https://api.upstox.com/v3/historical-candle'


def _retry_get(session, url, *, headers=None, attempts=4, sleep=time.sleep):
    for n in range(attempts):
        try:
            response = session.get(url, headers=headers, timeout=(10, 45))
        except requests.RequestException:
            if n == attempts - 1:
                raise
            sleep(min(2 ** n, 10))
            continue
        if response.status_code in (429, 500, 502, 503, 504) and n < attempts - 1:
            delay = response.headers.get('Retry-After', '')
            try:
                delay = max(.2, min(60, float(delay)))
            except ValueError:
                delay = min(2 ** n, 10)
            sleep(delay)
            continue
        response.raise_for_status()  # 401/403 are fatal; do not pretend no data.
        return response
    raise RuntimeError('Upstox retries exhausted')


def parse_universe(csv_text: str, contracts: list[dict]) -> pd.DataFrame:
    constituents = pd.read_csv(StringIO(csv_text)).rename(columns=lambda c: c.strip())
    if 'Symbol' not in constituents or len(constituents) != 50:
        raise ValueError('Official Nifty 50 list must contain exactly 50 stocks')
    symbols = constituents['Symbol'].astype(str).str.strip().str.upper().tolist()
    if len(set(symbols)) != 50:
        raise ValueError('Duplicate Nifty 50 constituents')
    equities = {}
    for c in contracts:
        if c.get('segment') != 'NSE_EQ' or c.get('instrument_type') != 'EQ':
            continue
        symbol = str(c.get('trading_symbol', '')).strip().upper()
        if symbol in symbols:
            equities.setdefault(symbol, []).append(c)
    rows = []
    for symbol in symbols:
        matches = equities.get(symbol, [])
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely map current Nifty 50 stock {symbol}: {len(matches)} matches')
        c = matches[0]
        isin, key = str(c.get('isin', '')), str(c.get('instrument_key', ''))
        if not isin.startswith('IN') or key != f'NSE_EQ|{isin}':
            raise ValueError(f'Invalid Upstox instrument identity for {symbol}')
        rows.append({'symbol': symbol, 'isin': isin, 'instrument_key': key,
                     'company': str(c.get('name', ''))})
    return pd.DataFrame(rows).sort_values('symbol').reset_index(drop=True)


def snapshot_universe(session, root: Path, *, asof: str, universe_csv: Path | None = None,
                      instrument_json: Path | None = None) -> pd.DataFrame:
    """Fetch fresh official list and BOD instruments, or explicit user-supplied files."""
    csv_bytes = universe_csv.read_bytes() if universe_csv else _retry_get(session, NIFTY50_URL).content
    if instrument_json:
        contents = instrument_json.read_bytes()
    else:
        contents = _retry_get(session, UPSTOX_NSE_INSTRUMENTS_URL).content
    if contents.startswith(b'\x1f\x8b'):
        contents = gzip.decompress(contents)
    universe = parse_universe(csv_bytes.decode('utf-8-sig'), json.loads(contents))
    dst = root / 'universes' / 'nifty50' / f'{asof}.csv'
    dst.parent.mkdir(parents=True, exist_ok=True)
    encoded = universe.to_csv(index=False).encode('utf-8')
    if dst.exists() and dst.read_bytes() != encoded:
        raise ValueError(f'Snapshot already exists and differs: {dst}; use a new --asof date')
    dst.write_bytes(encoded)
    provenance = {'snapshot_date': asof, 'current_constituents_only': True,
                  'historical_constituents_reconstructed': False,
                  'universe_url': str(universe_csv or NIFTY50_URL),
                  'instruments_url': str(instrument_json or UPSTOX_NSE_INSTRUMENTS_URL),
                  'sha256': hashlib.sha256(encoded).hexdigest(), 'count': len(universe)}
    (dst.parent / f'{asof}.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return universe


def parse_candles(candles, row, year, first, last):
    parsed = []
    for candle in candles:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            raise ValueError('Invalid Upstox V3 daily candle payload')
        ts = pd.Timestamp(candle[0]); d = ts.tz_convert('Asia/Kolkata').date() if ts.tz is not None else ts.date()
        if d.year != year or not (first <= d <= last):
            raise ValueError(f'Out-of-range session {d} returned for {row.symbol}')
        parsed.append({'date': d.isoformat(), 'symbol': row.symbol, 'isin': row.isin,
                       'instrument_key': row.instrument_key, 'open': candle[1], 'high': candle[2],
                       'low': candle[3], 'close': candle[4], 'volume': candle[5],
                       'open_interest': candle[6] if len(candle) > 6 else 0})
    if not parsed:
        return pd.DataFrame(columns=COLUMNS)
    return validate_bars(pd.DataFrame(parsed))


def candle_url(key: str, first: date, last: date) -> str:
    if first > last or (last - first).days > 3652:
        raise ValueError('Daily historical range exceeds ten years')
    return f'{API}/{quote(key, safe="")}/days/1/{last.isoformat()}/{first.isoformat()}'


def fetch_candles(session, token, row, year: int, first: date, last: date):
    if not token:
        raise ValueError('UPSTOX_ACCESS_TOKEN is required for authenticated historical candles')
    url = candle_url(row.instrument_key, first, last)
    res = _retry_get(session, url, headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'})
    payload = res.json()
    if payload.get('status') != 'success' or not isinstance(payload.get('data', {}).get('candles'), list):
        raise ValueError(f'Upstox response has no valid candles array for {row.symbol}/{year}')
    return parse_candles(payload['data']['candles'], row, year, first, last)


def collect(*, root, start: str, end: str, token: str | None = None,
            asof: str | None = None, universe_csv: str | None = None,
            instrument_json: str | None = None, session=None, pause: float = .12):
    root = data_root(root)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or first < date(2000, 1, 1) or last > date.today():
        raise ValueError('Invalid historical date range')
    if first != date(first.year,1,1) or last != date(last.year,12,31):
        raise ValueError('For resumable yearly partitions use full-year dates Jan 1 through Dec 31')
    token = token or os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not token:
        raise ValueError('Set UPSTOX_ACCESS_TOKEN locally (never commit the token); download not started')
    asof = asof or date.today().isoformat()
    session = session or requests.Session()
    universe = snapshot_universe(session, root, asof=asof,
                                  universe_csv=Path(universe_csv) if universe_csv else None,
                                  instrument_json=Path(instrument_json) if instrument_json else None)
    rows = []
    for item in universe.itertuples(index=False):
        for year in range(first.year, last.year+1):
            begin, finish = max(first, date(year, 1, 1)), min(last, date(year, 12, 31))
            path = bar_path(root, item.isin, year)
            if path.exists():
                existing = validate_bars(pd.read_csv(path))
                if not existing.empty and existing['isin'].eq(item.isin).all():
                    rows.append({'symbol':item.symbol,'year':year,'status':'cached','rows':len(existing),'error':''})
                    continue
            try:
                daily = fetch_candles(session, token, item, year, begin, finish)
                if not daily.empty:
                    save_year(root, daily, isin=item.isin, year=year)
                rows.append({'symbol':item.symbol,'year':year,'status':'downloaded' if len(daily) else 'no_data','rows':len(daily),'error':''})
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                # Never serialize request headers or bearer tokens into reports.
                code = getattr(getattr(exc, 'response', None), 'status_code', None)
                rows.append({'symbol':item.symbol,'year':year,'status':'error','rows':0,
                             'error':f'{type(exc).__name__} HTTP={code}' if code else type(exc).__name__})
                if code in (401, 403):  # stop instead of creating hundreds of identical failures
                    break
            if pause:
                time.sleep(pause)
        if rows and rows[-1]['status'] == 'error' and rows[-1]['error'].endswith(('HTTP=401','HTTP=403')):
            break
    report_dir = root/'reports'/'upstox_v3'; report_dir.mkdir(parents=True, exist_ok=True)
    name = f'nifty50_{asof}_{start}_{end}'
    status = pd.DataFrame(rows)
    status.to_csv(report_dir/f'{name}_status.csv', index=False)
    summary = {'start':start,'end':end,'snapshot':asof,'universe_size':len(universe),
               'requests_or_cache_checks':len(rows),'downloaded':int(status.status.eq('downloaded').sum()),
               'cached':int(status.status.eq('cached').sum()),'no_data':int(status.status.eq('no_data').sum()),
               'errors':int(status.status.eq('error').sum()),'data_root':str(root),
               'corporate_action_adjustment':'UNVERIFIED',
               'historical_membership':'CURRENT_NIFTY50_ONLY',
               'source':'Upstox Historical Candle V3 /days/1', 'live_execution':False}
    (report_dir/f'{name}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=False, help='Shared data folder OUTSIDE ai-swing')
    p.add_argument('--from-date', required=True); p.add_argument('--to-date', required=True)
    p.add_argument('--asof', help='Current Nifty50 snapshot date YYYY-MM-DD; omit for today')
    p.add_argument('--universe-csv', help='Optional offline Nifty50 official list CSV')
    p.add_argument('--instrument-json', help='Optional offline Upstox NSE JSON or JSON.gz')
    p.add_argument('--pause', type=float, default=.12)
    args=p.parse_args()
    report=collect(root=args.data_root,start=args.from_date,end=args.to_date,asof=args.asof,
                   universe_csv=args.universe_csv,instrument_json=args.instrument_json,pause=args.pause)
    print(json.dumps(report,indent=2))
    if report['errors']:
        p.exit(2, 'Some instrument/year downloads failed. Inspect the external status CSV.\n')


if __name__ == '__main__':
    main()