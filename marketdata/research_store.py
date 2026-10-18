"""Provider-neutral, versioned daily NSE OHLCV research data preparation.

Vendor files are immutable caches. Build a distinct, standardized research_ohlcv
series with per-partition provenance. No raw NSE data can silently mix with
adjusted historical OHLCV. This is offline and never places orders.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from marketdata.shared_store import (bar_path, data_root, load_universe, save_year,
                                     validate_bars, validate_universe_name)

CANONICAL = 'research_ohlcv'
BASE = 'yahoo_finance'


def _source_name(value: str) -> str:
    if value == BASE or (value.startswith('ext_') and 5 <= len(value) <= 48 and
                         value[4:].replace('-', '').replace('_', '').isalnum() and
                         value.lower() == value):
        return value
    raise ValueError('Source must be yahoo_finance or ext_<lowercase_provider_name>')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def import_external(*, root=None, csv_path, source: str, snapshot: str,
                    universe_name: str = 'nifty50', acknowledge_adjusted: bool = False) -> dict:
    """Normalize ONE stock/year of alternative adjusted OHLCV into its OWN cache.

    Input must have date,symbol,open,high,low,close,volume and match a frozen
    stock snapshot. The caller is responsible for corporate-action verification.
    Unknown adjustment provenance is not silently promoted into training.
    """
    if not acknowledge_adjusted:
        raise ValueError('External OHLCV must be adjusted; pass --confirm-adjusted-ohlcv to explicitly acknowledge')
    source = _source_name(source)
    if source == BASE:
        raise ValueError('Do not overwrite Yahoo cache with an alternate provider')
    root = data_root(root)
    universe = load_universe(root, snapshot, universe_name=universe_name)
    path = Path(csv_path)
    stock = pd.read_csv(path, dtype={'symbol':str, 'isin':str})
    mandatory = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
    if not mandatory.issubset(stock.columns) or stock.empty:
        raise ValueError('CSV must contain date,symbol,open,high,low,close,volume')
    if stock.symbol.isna().any() or stock.symbol.nunique() != 1:
        raise ValueError('Import exactly one NSE stock per CSV')
    symbol = stock.symbol.iloc[0]
    member = universe[universe.symbol.eq(symbol)]
    if len(member) != 1:
        raise ValueError(f'{symbol} is not part of the frozen {universe_name} snapshot')
    isin = member['isin'].iloc[0]
    if 'isin' in stock and (stock['isin'] != isin).any():
        raise ValueError('CSV ISIN does not match the frozen universe')
    stock['isin'] = isin
    checked = validate_bars(stock)
    if checked.date.dt.year.nunique() != 1:
        raise ValueError('Import exactly one complete-or-partial calendar year per file')
    year = int(checked.date.dt.year.iloc[0])
    target = save_year(root, checked, isin=isin, year=year, provider=source)
    metadata = {'source':source, 'source_file':str(path), 'source_input_sha256':_digest(path),
                'cache_sha256':_digest(target), 'snapshot':snapshot, 'universe_name':universe_name,
                'symbol':symbol, 'isin':isin, 'year':year, 'rows':len(checked),
                'price_adjustment':'external_provider_adjustment_not_independently_verified',
                'live_trading_enabled':False}
    meta_path = target.with_name(f'{year}.source.json')
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old['cache_sha256'] != metadata['cache_sha256']:
            raise ValueError('External data provenance conflict')
    else:
        _write_json(meta_path, metadata)
    return metadata


def build_view(*, root=None, snapshot: str, start: str, end: str,
               universe_name: str = 'nifty50', sources=('yahoo_finance',),
               refresh: bool = False) -> dict:
    """Freeze an independent ISIN/year research view with recorded vendor origin.

    The view is never mutated unless refresh=True; reruns reuse already selected
    partitions. Missing stock-years remain absent and auditable, not fabricated.
    """
    root = data_root(root)
    universe_name = validate_universe_name(universe_name)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (first.month, first.day) != (1, 1) or (last.month, last.day) != (12, 31):
        raise ValueError('Require full calendar years YYYY-01-01 to YYYY-12-31')
    sources = [_source_name(s) for s in sources]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Pass unique --sources in priority order')
    universe = load_universe(root, snapshot, universe_name=universe_name)
    status = []
    for item in universe.itertuples(index=False):
        for year in range(first.year, last.year + 1):
            dest = bar_path(root, item.isin, year, CANONICAL)
            record = {'symbol':item.symbol, 'isin':item.isin, 'year':year,
                      'status':'missing', 'source':'', 'rows':0, 'error':''}
            try:
                if dest.is_file() and not refresh:
                    rows = validate_bars(pd.read_csv(dest,dtype={'isin':str}))
                    if rows.symbol.iloc[0] != item.symbol or rows['isin'].iloc[0] != item.isin or not rows.date.dt.year.eq(year).all():
                        raise ValueError('Existing canonical stock/year identity mismatch')
                    provenance = dest.with_name(f'{year}.source.json')
                    if not provenance.is_file() or json.loads(provenance.read_text())['canonical_sha256'] != _digest(dest):
                        raise ValueError('Missing/corrupt provenance; canonical cache cannot be trusted')
                    record.update(status='cached', rows=len(rows), source=json.loads(provenance.read_text())['source'])
                else:
                    for source in sources:
                        origin = bar_path(root, item.isin, year, source)
                        if not origin.is_file():
                            continue
                        if source.startswith('ext_'):
                            meta = origin.with_name(f'{year}.source.json')
                            if not meta.is_file() or json.loads(meta.read_text()).get('cache_sha256') != _digest(origin):
                                raise ValueError(f'External source provenance missing or changed: {origin}')
                        data = validate_bars(pd.read_csv(origin, dtype={'isin':str}))
                        if (data.symbol.iloc[0] != item.symbol or data['isin'].iloc[0] != item.isin or
                                not data.date.dt.year.eq(year).all()):
                            raise ValueError(f'Source identity/year mismatch: {origin}')
                        save_year(root, data, isin=item.isin, year=year, provider=CANONICAL, replace=refresh)
                        _write_json(dest.with_name(f'{year}.source.json'), {
                            'source':source, 'source_path':str(origin), 'source_sha256':_digest(origin),
                            'canonical_sha256':_digest(dest), 'price_adjustment':'provider_adjusted_unverified',
                            'symbol':item.symbol, 'isin':item.isin, 'year':year, 'rows':len(data),
                            'universe_snapshot':snapshot, 'live_trading_enabled':False})
                        record.update(status='selected',source=source,rows=len(data))
                        break
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}'[:250])
            status.append(record)
    folder = root/'reports'/CANONICAL
    stem = f'{universe_name}_{snapshot}_{start}_{end}'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status).to_csv(folder/f'{stem}_status.csv',index=False)
    counts = pd.Series([x['status'] for x in status]).value_counts().to_dict()
    result = {'provider':CANONICAL,'universe_name':universe_name,'snapshot':snapshot,
              'start':start,'end':end,'universe_size':len(universe),
              'requests_or_cache_checks':len(status), 'sources_in_priority_order':sources,
              'selected':int(counts.get('selected',0)), 'cached':int(counts.get('cached',0)),
              'missing':int(counts.get('missing',0)), 'errors':int(counts.get('error',0)),
              'status_csv':str(folder/f'{stem}_status.csv'),
              'price_adjustment':'PROVIDER_ADJUSTED_UNVERIFIED',
              'live_trading_enabled':False}
    _write_json(folder/f'{stem}_summary.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',help='External shared market-data directory')
    p.add_argument('--snapshot',required=True,help='Frozen stock-universe date')
    p.add_argument('--universe-name',default='nifty50')
    commands = p.add_subparsers(dest='action',required=True)
    imported = commands.add_parser('import', help='Normalize alternative provider CSV, one stock/year')
    imported.add_argument('--csv',required=True)
    imported.add_argument('--source',required=True,help='e.g. ext_paid_vendor')
    imported.add_argument('--confirm-adjusted-ohlcv',action='store_true',
                          help='Assert CSV has adjusted daily OHLCV; requires independent corporate-action review')
    built = commands.add_parser('build', help='Build independent standardized research OHLCV')
    built.add_argument('--from-date',required=True)
    built.add_argument('--to-date',required=True)
    built.add_argument('--sources',nargs='+',default=['yahoo_finance'])
    built.add_argument('--refresh',action='store_true',help='Explicitly regenerate canonical data (changes research dataset)')
    args = p.parse_args()
    if args.action == 'import':
        out = import_external(root=args.data_root,csv_path=args.csv,source=args.source,
                              snapshot=args.snapshot,universe_name=args.universe_name,
                              acknowledge_adjusted=args.confirm_adjusted_ohlcv)
    else:
        out = build_view(root=args.data_root,snapshot=args.snapshot,universe_name=args.universe_name,
                         start=args.from_date,end=args.to_date,sources=args.sources,refresh=args.refresh)
    print(json.dumps(out,indent=2))
    if out.get('errors'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()