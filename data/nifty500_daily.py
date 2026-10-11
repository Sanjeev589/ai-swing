"""Download official NSE end-of-day open/close/volume for a current Nifty 500 snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO, BytesIO
import json
import gzip
import hashlib
from pathlib import Path
import time
import pandas as pd
import numpy as np
import requests

CONSTITUENTS_URL = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
ARCHIVE_URL = 'https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day}.csv'
COLUMNS = ['date', 'symbol', 'open', 'close', 'volume']


def validate_ocv(frame):
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError('Missing daily open/close/volume columns')
    frame = frame[COLUMNS].copy()
    if frame.empty:
        raise ValueError('No matching equity rows')
    if frame[['date', 'symbol']].isna().any().any():
        raise ValueError('Missing date or symbol')
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.date.isna().any() or frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Expected daily session dates')
    frame['symbol'] = frame.symbol.str.strip().str.upper()
    if frame.symbol.eq('').any() or frame.duplicated(['date', 'symbol']).any():
        raise ValueError('Empty symbol or duplicate symbol/date')
    for column in ['open', 'close', 'volume']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(frame[['open', 'close', 'volume']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite values')
    if (frame.close.le(0).any() or frame.open.lt(0).any() or ((frame.open == 0) & (frame.volume > 0)).any()
            or frame.volume.lt(0).any() or not frame.volume.eq(frame.volume.round()).all()):
        raise ValueError('Invalid prices or volume')
    frame['volume'] = frame.volume.astype('int64')
    return frame.sort_values(['date', 'symbol']).reset_index(drop=True)


def parse_bhavcopy(text, session, symbols):
    if isinstance(text, bytes):
        if text.startswith(b'PK'):
            frame = pd.read_excel(BytesIO(text), engine='openpyxl')
        else:
            frame = pd.read_csv(StringIO(text.decode('utf-8-sig')), skipinitialspace=True)
    else:
        frame = pd.read_csv(StringIO(text), skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    needed = {'SYMBOL', 'SERIES', 'DATE1', 'OPEN_PRICE', 'CLOSE_PRICE', 'TTL_TRD_QNTY'}
    if not needed.issubset(frame.columns):
        raise ValueError('Unexpected NSE archive schema')
    frame['SYMBOL'] = frame.SYMBOL.str.strip()
    frame = frame[frame.SERIES.str.strip().isin(['EQ', 'BE', 'BZ', 'SM', 'ST', 'RR']) & frame.SYMBOL.isin(symbols)].copy()
    frame['date'] = pd.to_datetime(frame.DATE1.str.strip(), format='%d-%b-%Y', errors='raise')
    if not frame.date.eq(pd.Timestamp(session)).all():
        raise ValueError('Archive date does not match requested session')
    frame = frame.rename(columns={'SYMBOL': 'symbol', 'OPEN_PRICE': 'open',
                                  'CLOSE_PRICE': 'close', 'TTL_TRD_QNTY': 'volume'})
    return validate_ocv(frame)


def download_dataset(start, end, output, *, workers=6, constituents_file=None, cache='data/local/nse_daily_cache'):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end or workers < 1 or workers > 32:
        raise ValueError('Invalid dates or workers (1..32)')
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if constituents_file:
        constituent_text = Path(constituents_file).read_text(encoding='utf-8')
    else:
        response = requests.get(CONSTITUENTS_URL, timeout=30)
        response.raise_for_status()
        constituent_text = response.text
    constituents = pd.read_csv(StringIO(constituent_text))
    if not {'Symbol', 'ISIN Code'}.issubset(constituents.columns):
        raise ValueError('Invalid official constituent snapshot')
    if not 490 <= len(constituents) <= 510 or constituents.Symbol.duplicated().any():
        raise ValueError('Unexpected constituent count or duplicates')
    symbols = set(constituents.Symbol.str.strip())
    snapshot_hash = hashlib.sha256('\n'.join(sorted(symbols)).encode()).hexdigest()
    snapshot_file = cache / '.snapshot.sha256'
    if snapshot_file.exists() and snapshot_file.read_text().strip() != snapshot_hash:
        raise ValueError('Constituents changed; use a new cache directory')
    snapshot_file.write_text(snapshot_hash, encoding='utf-8')
    constituents.to_csv(output / 'constituents.csv', index=False)
    audit_path = output / 'archive_status.csv'
    known_statuses = {}
    if audit_path.exists():
        known_statuses = {row['date']: row for row in pd.read_csv(audit_path).to_dict('records')}
    dates = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')
    # Include weekends: NSE may hold special sessions. Archive availability is authoritative.
    def fetch(session):
        target = cache / f'{session:%Y-%m-%d}.csv'
        if target.exists():
            data = validate_ocv(pd.read_csv(target))
            if not data.symbol.isin(symbols).all() or not data.date.eq(session).all():
                raise ValueError('Cached snapshot mismatch; use a new cache directory')
            return {'date': f'{session:%Y-%m-%d}', 'status': 'available', 'rows': len(data)}, data
        known = known_statuses.get(f'{session:%Y-%m-%d}')
        if known and known['status'] in {'archive_not_found', 'archive_date_mismatch'}:
            return known, None
        url = ARCHIVE_URL.format(day=f'{session:%d%m%Y}')
        error = ''
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=20)
                if response.status_code == 404:
                    return {'date': f'{session:%Y-%m-%d}', 'status': 'archive_not_found', 'rows': 0}, None
                response.raise_for_status()
                try:
                    data = parse_bhavcopy(response.content, session, symbols)
                except ValueError as exc:
                    if 'date does not match' in str(exc):
                        return {'date': f'{session:%Y-%m-%d}', 'status': 'archive_date_mismatch', 'rows': 0}, None
                    raise
                temporary = target.with_suffix('.tmp')
                data.to_csv(temporary, index=False, date_format='%Y-%m-%d')
                temporary.replace(target)
                return {'date': f'{session:%Y-%m-%d}', 'status': 'available', 'rows': len(data)}, data
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:300]
                if attempt < 2:
                    time.sleep(attempt + 1)
        return {'date': f'{session:%Y-%m-%d}', 'status': 'error', 'rows': 0, 'error': error}, None
    statuses, frames = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, session) for session in dates]
        for count, future in enumerate(as_completed(futures), 1):
            status, frame = future.result()
            statuses.append(status)
            if status['status'] == 'error':
                print(f"ERROR {status['date']}: {status.get('error', '')}", flush=True)
            if frame is not None:
                frames.append(frame)
            if count % 100 == 0 or count == len(futures):
                print(f'Checked {count}/{len(futures)} dates; sessions={len(frames)}; errors={sum(s["status"] == "error" for s in statuses)}', flush=True)
    statuses = pd.DataFrame(statuses).sort_values('date')
    statuses.to_csv(output / 'archive_status.csv', index=False)
    if not frames:
        raise ValueError('No historical sessions downloaded; inspect archive_status.csv')
    combined = validate_ocv(pd.concat(frames, ignore_index=True))
    coverage = combined.groupby('symbol').agg(first_date=('date', 'min'), last_date=('date', 'max'), rows=('date', 'size')).reindex(sorted(symbols))
    coverage['rows'] = coverage.rows.fillna(0).astype(int)
    coverage['missing_from_available_sessions'] = combined.date.nunique() - coverage.rows
    coverage.reset_index().to_csv(output / 'coverage.csv', index=False, date_format='%Y-%m-%d')
    errors = int(statuses.status.eq('error').sum())
    manifest = {'source': 'NSE security-wise daily bhavcopy', 'archive_url_template': ARCHIVE_URL,
                'constituents_source': CONSTITUENTS_URL, 'snapshot_retrieved_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                'start_inclusive': f'{start:%Y-%m-%d}', 'end_exclusive': f'{end:%Y-%m-%d}',
                'frequency': 'daily', 'series_filter': ['EQ', 'BE', 'BZ', 'SM', 'ST', 'RR'], 'columns': COLUMNS, 'currency': 'INR',
                'volume_definition': 'TTL_TRD_QNTY (total traded shares/units for the session)',
                'price_adjustment': 'As-published NSE open/close; not adjusted for corporate actions',
                'universe': 'Current constituent snapshot, not historical point-in-time membership',
                'constituents': len(symbols), 'symbols_with_data': combined.symbol.nunique(),
                'available_sessions': combined.date.nunique(), 'rows': len(combined),
                'first_date': str(combined.date.min().date()), 'last_date': str(combined.date.max().date()),
                'archive_not_found_dates': int(statuses.status.eq('archive_not_found').sum()),
                'archive_date_mismatch_dates': int(statuses.status.eq('archive_date_mismatch').sum()),
                'download_errors': errors, 'download_status': 'complete_requests' if errors == 0 else 'incomplete',
                'notes': ['404 dates are unavailable archives, not independently verified holidays.',
                          'Short coverage may reflect IPOs, symbol changes, suspensions, or unavailable archives; no rows are fabricated.',
                          'Raw open/close/volume dataset cannot be passed to the existing OHLCV feature loader; ATR requires high/low.']}
    if errors == 0:
        manifest['files'] = []
        for year, annual in combined.groupby(combined.date.dt.year):
            path = output / f'daily_{year}.csv.gz'
            payload = annual.to_csv(index=False, date_format='%Y-%m-%d', float_format='%.2f').encode('utf-8')
            path.write_bytes(gzip.compress(payload, mtime=0))
            manifest['files'].append({'path': path.name, 'rows': len(annual),
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if errors:
        raise RuntimeError(f'{errors} download errors; dataset not finalized. Rerun to resume cached sessions.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2019-10-11')
    parser.add_argument('--end', default='2026-10-11', help='Exclusive date')
    parser.add_argument('--output', default='data/nifty500')
    parser.add_argument('--cache', default='data/local/nse_daily_cache')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--constituents-file')
    args = parser.parse_args()
    print(json.dumps(download_dataset(args.start, args.end, args.output, workers=args.workers,
                                      cache=args.cache, constituents_file=args.constituents_file), indent=2))


if __name__ == '__main__':
    main()
