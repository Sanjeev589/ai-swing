"""Official NSE daily OHLCV and Nifty 500 closes for indicator generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from zipfile import ZipFile
from pathlib import Path
import json
import gzip
import hashlib
import time
import numpy as np
import pandas as pd
import requests

STOCK_URL = 'https://nsearchives.nseindia.com/content/historical/EQUITIES/{year}/{month}/cm{day}bhav.csv.zip'
INDEX_URL = 'https://nsearchives.nseindia.com/content/indices/ind_close_all_{day}.csv'
SERIES = ['EQ', 'BE', 'BZ', 'SM', 'ST', 'RR']
PRICE_COLUMNS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']


def read_benchmark(content, session):
    frame = pd.read_csv(BytesIO(content), skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    if not {'Index Name', 'Index Date', 'Closing Index Value'}.issubset(frame.columns):
        raise ValueError('Unexpected index snapshot schema')
    names = frame['Index Name'].str.strip().str.upper().str.replace(' ', '', regex=False)
    row = frame[names.isin(['NIFTY500', 'CNX500'])]
    if len(row) != 1:
        raise ValueError('Expected exactly one Nifty 500 price index row')
    date = pd.to_datetime(row['Index Date'].iloc[0].strip(), format='%d-%m-%Y')
    if date != pd.Timestamp(session):
        raise ValueError('Index date mismatch')
    close = float(row['Closing Index Value'].iloc[0])
    if not np.isfinite(close) or close <= 0:
        raise ValueError('Invalid index close')
    return {'date': date, 'nifty500_close': close}


def read_stock_zip(content, session, symbols):
    with ZipFile(BytesIO(content)) as archive:
        files = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(files) != 1:
            raise ValueError('Expected one bhavcopy CSV in archive')
        frame = pd.read_csv(archive.open(files[0]), skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    required = {'SYMBOL', 'SERIES', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TIMESTAMP'}
    if not required.issubset(frame.columns):
        raise ValueError('Unexpected stock bhavcopy schema')
    frame = frame[frame.SERIES.str.strip().isin(SERIES) & frame.SYMBOL.str.strip().isin(symbols)].copy()
    timestamps = frame.TIMESTAMP.str.strip()
    dates = pd.to_datetime(timestamps, format='%d-%b-%Y', errors='coerce')
    frame['date'] = dates.fillna(pd.to_datetime(timestamps, format='%d-%b-%y', errors='coerce'))
    if frame.date.isna().any():
        raise ValueError('Invalid stock archive timestamp')
    if not frame.date.eq(pd.Timestamp(session)).all():
        raise ValueError('Stock archive date mismatch')
    frame = frame.rename(columns={'SYMBOL': 'symbol', 'OPEN': 'open', 'HIGH': 'high',
                                  'LOW': 'low', 'CLOSE': 'close', 'TOTTRDQTY': 'volume'})
    return validate_prices(frame)


def validate_prices(frame):
    if not set(PRICE_COLUMNS).issubset(frame.columns) or frame.empty:
        raise ValueError('Missing OHLCV inputs')
    frame = frame[PRICE_COLUMNS].copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    frame['symbol'] = frame.symbol.str.strip().str.upper()
    if frame[['date', 'symbol']].isna().any().any() or frame.symbol.eq('').any() or frame.duplicated(['date', 'symbol']).any():
        raise ValueError('Missing keys or duplicate quotes')
    if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Expected daily session dates')
    numeric = ['open', 'high', 'low', 'close', 'volume']
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite OHLCV inputs')
    if frame[numeric].lt(0).any().any() or frame.close.le(0).any() or not frame.volume.eq(frame.volume.round()).all():
        raise ValueError('Invalid OHLCV inputs')
    active = frame.volume.gt(0)
    if frame.loc[active, ['open', 'high', 'low']].le(0).any().any():
        raise ValueError('Nonpositive active-session prices')
    if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)) & active).any() or ((frame.low > frame[['open', 'close', 'high']].min(axis=1)) & active).any():
        raise ValueError('Inconsistent OHLC ranges')
    frame['volume'] = frame.volume.astype('int64')
    return frame.sort_values(['date', 'symbol']).reset_index(drop=True)


def collect_inputs(output='data/feature_inputs', start='2017-12-01', end='2023-01-01', workers=16,
                   constituents='data/nifty500/constituents.csv', cache='data/local/feature_input_cache'):
    if not 1 <= workers <= 32 or pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError('Invalid workers or date range')
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(constituents)
    symbols = set(universe.Symbol)
    fingerprint = hashlib.sha256('\n'.join(sorted(symbols)).encode()).hexdigest()
    key = cache / '.universe.sha256'
    if key.exists() and key.read_text().strip() != fingerprint:
        raise ValueError('Universe changed; select a new cache')
    key.write_text(fingerprint)
    audit = output / 'download_status.csv'
    known = {r['date']: r for r in pd.read_csv(audit).to_dict('records')} if audit.exists() else {}
    def fetch(date):
        stem = f'{date:%Y-%m-%d}'
        stock_path, index_path = cache / f'{stem}.csv.gz', cache / f'{stem}.json'
        if stock_path.exists() and index_path.exists():
            prices = validate_prices(pd.read_csv(stock_path))
            benchmark = json.loads(index_path.read_text()); benchmark['date'] = pd.Timestamp(benchmark['date'])
            if not prices.date.eq(date).all() or benchmark['date'] != date:
                raise ValueError('Cache date mismatch')
            return {'date': stem, 'status': 'available', 'rows': len(prices)}, prices, benchmark
        if stem in known and known[stem]['status'] in {'index_not_found', 'index_date_mismatch'}:
            return known[stem], None, None
        error = ''
        for attempt in range(3):
            try:
                index_response = requests.get(INDEX_URL.format(day=f'{date:%d%m%Y}'), timeout=20)
                if index_response.status_code == 404:
                    return {'date': stem, 'status': 'index_not_found', 'rows': 0}, None, None
                index_response.raise_for_status()
                try:
                    benchmark = read_benchmark(index_response.content, date)
                except ValueError as exc:
                    if 'date mismatch' in str(exc):
                        return {'date': stem, 'status': 'index_date_mismatch', 'rows': 0}, None, None
                    raise
                response = requests.get(STOCK_URL.format(year=date.year, month=f'{date:%b}'.upper(), day=f'{date:%d%b%Y}'.upper()), timeout=20)
                response.raise_for_status()
                prices = read_stock_zip(response.content, date, symbols)
                prices.to_csv(stock_path, index=False, date_format='%Y-%m-%d', compression='gzip')
                index_path.write_text(json.dumps({'date': stem, 'nifty500_close': benchmark['nifty500_close']}))
                return {'date': stem, 'status': 'available', 'rows': len(prices)}, prices, benchmark
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:250]
                if attempt < 2: time.sleep(attempt + 1)
        return {'date': stem, 'status': 'error', 'error': error, 'rows': 0}, None, None
    dates = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq='D')
    statuses, frames, benchmarks = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, date) for date in dates]
        for count, future in enumerate(as_completed(futures), 1):
            status, prices, benchmark = future.result(); statuses.append(status)
            if prices is not None: frames.append(prices); benchmarks.append(benchmark)
            if status['status'] == 'error': print('ERROR', status, flush=True)
            if count % 100 == 0 or count == len(dates):
                print(f'Inputs {count}/{len(dates)} dates; sessions={len(frames)}; errors={sum(s["status"] == "error" for s in statuses)}', flush=True)
    statuses = pd.DataFrame(statuses).sort_values('date'); statuses.to_csv(audit, index=False)
    if statuses.status.eq('error').any() or not frames:
        raise RuntimeError('Inputs incomplete; rerun to resume from the validated cache')
    prices = validate_prices(pd.concat(frames, ignore_index=True))
    benchmark = pd.DataFrame(benchmarks).sort_values('date').reset_index(drop=True)
    if benchmark.date.duplicated().any(): raise ValueError('Duplicate benchmark dates')
    # Check overlapping O/C/V quotes against the original published dataset.
    existing = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = Path('data/nifty500') / f'daily_{year}.csv.gz'
        if path.exists(): existing.append(pd.read_csv(path, parse_dates=['date']))
    if existing:
        original = pd.concat(existing, ignore_index=True)
        original = original[(original.date >= pd.Timestamp(start)) & (original.date < pd.Timestamp(end))]
        matched = original.merge(prices, on=['date', 'symbol'], how='left', suffixes=('_original', '_support'), validate='one_to_one')
        for field in ['open', 'close', 'volume']:
            if matched[field + '_support'].isna().any() or not np.allclose(matched[field + '_original'], matched[field + '_support'], atol=1e-8, rtol=0):
                raise ValueError(f'Supporting archive disagrees with original {field} data')
    files = []
    for year, annual in prices.groupby(prices.date.dt.year):
        path = output / f'ohlcv_{year}.csv.gz'
        path.write_bytes(gzip.compress(annual.to_csv(index=False, date_format='%Y-%m-%d', float_format='%.2f').encode(), mtime=0))
        files.append({'path': path.name, 'rows': len(annual), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    index_path = output / 'nifty500_index.csv'
    benchmark.to_csv(index_path, index=False, date_format='%Y-%m-%d')
    files.append({'path': index_path.name, 'rows': len(benchmark),
                  'sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()})
    manifest = {'stock_source': STOCK_URL, 'index_source': INDEX_URL, 'constituents': constituents,
                'start_inclusive': start, 'end_exclusive': end, 'rows': len(prices),
                'sessions': len(benchmark), 'price_adjustment': 'None; as-published NSE prices',
                'index_type': 'Nifty 500 price index, not total return index', 'files': files,
                'retrieved_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/feature_inputs')
    parser.add_argument('--start', default='2017-12-01'); parser.add_argument('--end', default='2023-01-01')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(collect_inputs(**vars(args)), indent=2))
