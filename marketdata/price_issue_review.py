"""Generate an editable price-issue review CSV, then publish verified corrections in a NEW version.

Scan only marks suspects; no automated corporate-action inference. Replacements require
an explicit review, dated evidence/notes, and full OHLCV. Neither command modifies
source history, an existing research version, or live trading settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from marketdata.shared_store import bar_path, data_root, load_universe, validate_bars

ISSUE_COLUMNS = [
    'issue_id', 'symbol', 'isin', 'date', 'year', 'suspected_issues', 'priority',
    'source_provider', 'prior_source', 'source_price_basis', 'source_partition_sha256',
    'prior_date', 'prior_close', 'observed_open', 'observed_high', 'observed_low',
    'observed_close', 'observed_volume', 'close_change_pct', 'overnight_gap_pct',
    'calendar_gap_days',
    # The following fields are the ONLY fields reviewers are expected to edit.
    'verification_result', 'verification_notes', 'reference_url', 'reviewed_by',
    'correction_action', 'corrected_open', 'corrected_high', 'corrected_low',
    'corrected_close', 'corrected_volume',
]
OBSERVED_FIELDS = ('open', 'high', 'low', 'close', 'volume')
PERMITTED_RESULTS = {'pending', 'verified_issue', 'false_positive', 'unresolved'}
PERMITTED_ACTIONS = {'none', 'keep_original', 'replace_bar', 'exclude_stock_year'}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _issue_id(isin: str, date: str, digest: str) -> str:
    return hashlib.sha256(f'{isin}|{date}|{digest}'.encode()).hexdigest()[:24]


def _provider(s: str) -> str:
    if not re.fullmatch(r'research_v[a-z0-9][a-z0-9_-]{0,30}', s):
        raise ValueError('Only versioned research_v* datasets are supported')
    return s


def _partition(root: Path, provider: str, symbol: str, isin: str, year: int):
    path = bar_path(root, isin, year, provider)
    if not path.is_file():
        raise FileNotFoundError(path)
    meta_file = path.with_name(f'{year}.source.json')
    if not meta_file.is_file():
        raise ValueError(f'Missing research provenance: {meta_file}')
    metadata = json.loads(meta_file.read_text(encoding='utf-8'))
    if (metadata.get('canonical_sha256') != _hash(path) or
        metadata.get('symbol') != symbol or metadata.get('isin') != isin or
        metadata.get('year') != year):
        raise ValueError(f'Checksum or identity mismatch in: {path}')
    candles = validate_bars(pd.read_csv(path, dtype={'isin': str, 'symbol': str}), symbol=symbol)
    if not candles.date.dt.year.eq(year).all() or candles['isin'].iloc[0] != isin:
        raise ValueError(f'Data identity/year mismatch in: {path}')
    return path, metadata, candles


def _root_and_universe(root, snapshot, universe_name):
    root = data_root(root)
    universe = load_universe(root, snapshot, universe_name=universe_name)
    return root, universe


def scan(*, root=None, provider='research_v3', snapshot='2026-10-18', universe_name='nifty50',
         start='2000-01-01', end='2019-12-31', close_pct=20., overnight_pct=12.,
         source_switch_pct=10., gap_days=14, output=None):
    """One issue row per suspicious stock-session, including all source switches."""
    provider = _provider(provider)
    root, universe = _root_and_universe(root, snapshot, universe_name)
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a > b or any(not np.isfinite(x) or x <= 0 for x in (close_pct, overnight_pct, source_switch_pct)) or gap_days < 4:
        raise ValueError('Invalid date/threshold')
    target = Path(output) if output else (root/'reports'/'price_issue_review'/f'{provider}_{universe_name}_{snapshot}_issues.csv')
    if target.exists():
        raise FileExistsError(f'Review CSV exists; not overwriting user verification work: {target}')
    records = []
    for stock in universe.itertuples(index=False):
        fragments = []
        for year in range(a.year, b.year + 1):
            path = bar_path(root, stock.isin, year, provider)
            if not path.exists():
                continue
            _, metadata, frame = _partition(root, provider, stock.symbol, stock.isin, year)
            frame = frame.copy()
            frame['source_provider'] = metadata.get('source', 'unknown')
            frame['source_price_basis'] = metadata.get('price_basis', 'unknown')
            frame['source_partition_sha256'] = _hash(path)
            fragments.append(frame)
        if not fragments:
            continue
        combined = pd.concat(fragments, ignore_index=True).sort_values('date').reset_index(drop=True)
        prev = None
        for entry in combined.itertuples(index=False):
            now_date = entry.date.strftime('%Y-%m-%d')
            if a <= entry.date <= b and prev is not None:
                pct = 100. * (float(entry.close)/float(prev.close) - 1.)
                gap_pct = 100. * (float(entry.open)/float(prev.close) - 1.)
                days = int((entry.date-prev.date).days)
                switched = entry.source_provider != prev.source_provider
                basis_switch = entry.source_price_basis != prev.source_price_basis
                tags = []
                if abs(pct) >= close_pct: tags.append('large_close_change')
                if abs(gap_pct) >= overnight_pct: tags.append('large_open_gap')
                if days > gap_days: tags.append('long_calendar_gap')
                if switched: tags.append('source_switch')
                if basis_switch: tags.append('price_basis_switch')
                if tags:
                    high = basis_switch or abs(pct) >= close_pct or (switched and abs(pct)>=source_switch_pct)
                    item = {
                        'issue_id':_issue_id(stock.isin, now_date, entry.source_partition_sha256),
                        'symbol':stock.symbol,'isin':stock.isin,'date':now_date,'year':entry.date.year,
                        'suspected_issues':';'.join(tags),'priority':'HIGH' if high else 'REVIEW',
                        'source_provider':entry.source_provider,'prior_source':prev.source_provider,
                        'source_price_basis':entry.source_price_basis,
                        'source_partition_sha256':entry.source_partition_sha256,
                        'prior_date':prev.date.strftime('%Y-%m-%d'),'prior_close':prev.close,
                        'observed_open':entry.open,'observed_high':entry.high,
                        'observed_low':entry.low,'observed_close':entry.close,
                        'observed_volume':entry.volume,'close_change_pct':round(pct,4),
                        'overnight_gap_pct':round(gap_pct,4),'calendar_gap_days':days,
                        'verification_result':'pending','verification_notes':'','reference_url':'',
                        'reviewed_by':'','correction_action':'none',
                        **{f'corrected_{c}':'' for c in OBSERVED_FIELDS}}
                    records.append(item)
            prev = entry
    records.sort(key=lambda r:(r['priority']!='HIGH',r['symbol'],r['date']))
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=ISSUE_COLUMNS).to_csv(target, index=False)
    return {'input_provider':provider,'snapshot':snapshot,'review_file':str(target),
            'issues':len(records),'high_priority':sum(r['priority']=='HIGH' for r in records),
            'source_modified':False}


def _review_corrections(root: Path, parent: str, universe: pd.DataFrame, review: Path):
    data = pd.read_csv(review, dtype=str, keep_default_na=False)
    if not set(ISSUE_COLUMNS).issubset(data.columns):
        raise ValueError('Review CSV missing required columns: ' + ', '.join(sorted(set(ISSUE_COLUMNS)-set(data.columns))))
    if data.issue_id.duplicated().any() or data.duplicated(['isin','date']).any():
        raise ValueError('Duplicate issue_id or stock/date in review CSV')
    lookup = {(r.symbol,r.isin) for r in universe.itertuples(index=False)}
    changes, exclusions = {}, set()
    partition_cache = {}
    reviewed = 0
    for r in data.itertuples(index=False):
        key = (r.symbol, r.isin)
        if key not in lookup:
            raise ValueError(f'Unknown security in review: {key}')
        date = pd.Timestamp(r.date)
        if date.strftime('%Y-%m-%d') != r.date or int(r.year) != date.year:
            raise ValueError(f'Invalid date/year in issue {r.issue_id}')
        cache_key = (r.symbol, r.isin, date.year)
        if cache_key not in partition_cache:
            part, meta, bars = _partition(root, parent, r.symbol, r.isin, date.year)
            partition_cache[cache_key] = (part, meta, bars, _hash(part))
        part, meta, bars, digest = partition_cache[cache_key]
        if digest != r.source_partition_sha256 or _issue_id(r.isin,r.date,digest)!=r.issue_id:
            raise ValueError(f'Stale or edited source identity in issue {r.issue_id}')
        found = bars.loc[bars.date.eq(date)]
        if len(found) != 1:
            raise ValueError(f'Original date absent: {r.issue_id}')
        original=found.iloc[0]
        if r.source_provider != meta.get('source') or r.source_price_basis != meta.get('price_basis'):
            raise ValueError(f'Source identity changed in issue {r.issue_id}')
        for field in OBSERVED_FIELDS:
            if not np.isclose(float(getattr(r,'observed_'+field)), float(original[field]), rtol=1e-8, atol=1e-8):
                raise ValueError(f'Observed {field} differs from original: {r.issue_id}')
        if r.verification_result not in PERMITTED_RESULTS or r.correction_action not in PERMITTED_ACTIONS:
            raise ValueError(f'Unknown verification result/action in {r.issue_id}')
        if r.correction_action == 'none':
            continue
        if r.correction_action == 'keep_original':
            if r.verification_result != 'false_positive':
                raise ValueError(f'keep_original needs false_positive verification: {r.issue_id}')
            reviewed += 1
            continue
        if r.verification_result != 'verified_issue' or not r.verification_notes.strip() or not r.reference_url.strip():
            raise ValueError(f'Correction requires verified_issue, verification_notes and reference_url: {r.issue_id}')
        reviewed += 1
        stock_year = (r.isin, date.year)
        if r.correction_action == 'exclude_stock_year':
            exclusions.add(stock_year)
            continue
        values = {field:float(getattr(r,'corrected_'+field)) for field in OBSERVED_FIELDS}
        if any(not np.isfinite(v) for v in values.values()):
            raise ValueError(f'Non-finite correction in {r.issue_id}')
        if not any(not np.isclose(values[f],float(original[f]),atol=1e-8,rtol=1e-8) for f in OBSERVED_FIELDS):
            raise ValueError(f'Replacement matches original in {r.issue_id}')
        changes[(r.isin,r.date)] = (values,{'issue_id':r.issue_id,
            'verification_notes':r.verification_notes,'reference_url':r.reference_url,
            'reviewed_by':r.reviewed_by})
    if any((isin,int(date[:4])) in exclusions for isin,date in changes):
        raise ValueError('Cannot replace a bar while excluding its entire stock-year')
    return changes, exclusions, reviewed, len(data)


def apply(*, root=None, parent='research_v3', version='research_v4', snapshot='2026-10-18',
          universe_name='nifty50', review_file, dry_run=False):
    """Copy parent to a fresh version and apply only evidence-backed verified changes."""
    parent, version = _provider(parent), _provider(version)
    if parent == version:
        raise ValueError('Corrections must publish to a NEW research_v* version')
    root, universe = _root_and_universe(root, snapshot, universe_name)
    review_file = Path(review_file)
    if not review_file.is_file():
        raise FileNotFoundError(review_file)
    changes, excluded, reviewed, n_issues = _review_corrections(root,parent,universe,review_file)
    if not changes and not excluded:
        raise ValueError('No verified corrections to apply: review entries must select replace_bar or exclude_stock_year')
    target_dir = root/'candles'/version
    reports_dir = root/'reports'/version
    if target_dir.exists() or reports_dir.exists():
        raise FileExistsError(f'Version {version} exists. Choose a new version; original research data is immutable')
    partition_records=[]
    for member in universe.itertuples(index=False):
        for year in range(2000,2020):
            orig=bar_path(root,member.isin,year,parent)
            if not orig.is_file():
                continue
            old_path, old_meta, candles = _partition(root,parent,member.symbol,member.isin,year)
            if (member.isin,year) in excluded:
                continue
            corrections = [(day, values, provenance) for (isin,day),(values,provenance) in changes.items() if isin==member.isin and int(day[:4])==year]
            if corrections:
                candles=candles.copy()
                for day,values,info in corrections:
                    sel=candles.date.eq(pd.Timestamp(day))
                    if sel.sum()!=1:
                        raise ValueError(f'Correction target date missing: {member.symbol}/{day}')
                    for f,val in values.items():
                        candles.loc[sel,f]=val
                candles=validate_bars(candles,symbol=member.symbol)
            partition_records.append((member,year,old_path,old_meta,candles,corrections))
    all_changed_dates={(isin,day) for isin,day in changes}
    found_changes={(item.isin,day) for item,year,old,meta,candles,corrections in partition_records for day,_,_ in corrections}
    if all_changed_dates != found_changes:
        raise ValueError('Some correction rows point to excluded or missing partitions')
    review_sha=_hash(review_file)
    status = {'parent':parent,'provider':version,'snapshot':snapshot,'universe_name':universe_name,
              'review_file':str(review_file),'review_sha256':review_sha,
              'issue_rows':n_issues,'reviewed_rows':reviewed,
              'replaced_bars':len(changes),'excluded_stock_years':len(excluded),
              'selected_stock_years':len(partition_records),'dry_run':bool(dry_run),
              'source_data_modified':False,'live_execution_enabled':False}
    if dry_run:
        return status
    # Verify parent audits and prepare the new reports before publishing anything.
    parent_audits = {}
    for first,last in [('2000-01-01','2016-12-31'),('2017-01-01','2019-12-31')]:
        name=f'{universe_name}_{snapshot}_{first}_{last}_summary.json'
        audit_path=root/'reports'/parent/name
        if not audit_path.is_file():
            raise FileNotFoundError(f'Missing parent research audit: {audit_path}')
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if (audit.get('provider')!=parent or audit.get('snapshot')!=snapshot or
            audit.get('universe_name')!=universe_name or audit.get('pending')!=0 or
            audit.get('errors')!=0 or not audit.get('manifest_sha256')):
            raise ValueError(f'Invalid parent research audit: {audit_path}')
        parent_audits[(first,last)]=(name,audit)
    if len({v[1]['manifest_sha256'] for v in parent_audits.values()})!=1:
        raise ValueError('Parent train/holdout audits have different manifests')
    candle_root=root/'candles'; candle_root.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{version}-staging-',dir=candle_root))
    try:
        for member,year,old,metadata,candles,corrections in partition_records:
            dest=stage/'NSE_EQ'/'1d'/member.isin/f'{year}.csv.gz'
            dest.parent.mkdir(parents=True,exist_ok=True)
            if corrections:
                # Same canonical GZIP writer as shared_store.save_year; stage is not a provider.
                import gzip
                with dest.open('wb') as handle:
                    with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0) as stream:
                        stream.write(candles.to_csv(index=False,date_format='%Y-%m-%d').encode('utf-8'))
            else:
                shutil.copyfile(old,dest)
            latest=dict(metadata)
            latest.update({'version':version[len('research_'):],
                'canonical_sha256':_hash(dest),'parent_provider':parent,
                'parent_canonical_sha256':_hash(old),
                'price_issue_review_sha256':review_sha,
                'verified_bar_corrections':[dict(day=day,**info) for day,values,info in corrections],
                'live_trading_enabled':False})
            dest.with_name(f'{year}.source.json').write_text(json.dumps(latest,indent=2)+'\n',encoding='utf-8')
        stage.rename(target_dir)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    reports_dir.mkdir(parents=True,exist_ok=False)
    # The model requires two period-specific audits with the SAME review hash.
    for (first,last),(name,parent_audit) in parent_audits.items():
        audit=dict(parent_audit)
        audit.update({'provider':version,'version':version[len('research_'):],
            'manifest_sha256':review_sha,'parent_manifest_sha256':audit.get('manifest_sha256'),
            'parent_provider':parent,'price_issue_review_file':str(review_file),
            'price_issue_review_sha256':review_sha,'live_execution_enabled':False,
            'errors':0,'pending':0,
            'selected':sum(first <= f'{year}-01-01' <= last for _,year,*_ in partition_records),
            'excluded':sum(first <= f'{year}-01-01' <= last for item,year in excluded),
            'verified_price_repairs':sum(first <= day <= last for (isin,day) in changes)})
        (reports_dir/name).write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (reports_dir/'price_issue_review_result.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root')
    p.add_argument('--snapshot',default='2026-10-18')
    p.add_argument('--universe-name',default='nifty50')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan',help='Export CSV of suspected daily price/source issues')
    s.add_argument('--provider',default='research_v3')
    s.add_argument('--start',default='2000-01-01')
    s.add_argument('--end',default='2019-12-31')
    s.add_argument('--output')
    a=sub.add_parser('apply',help='Preflight and publish reviewed corrections into NEW research_v*')
    a.add_argument('--parent',default='research_v3')
    a.add_argument('--version',default='research_v4')
    a.add_argument('--review-file',required=True)
    a.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    kws={'root':args.data_root,'snapshot':args.snapshot,'universe_name':args.universe_name}
    result=scan(**kws,provider=args.provider,start=args.start,end=args.end,output=args.output) if args.command=='scan' else apply(**kws,parent=args.parent,version=args.version,review_file=args.review_file,dry_run=args.dry_run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()