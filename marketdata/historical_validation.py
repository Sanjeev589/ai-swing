"""Review historical NSE identity/listings and adjusted-price suitability before modeling.

No external network calls. The operator must verify and approve source identity and
corporate-action adjustments using independent evidence; Kaggle raw is NEVER model input.
The immutable, versioned selected price data live under research_<version>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.shared_store import bar_path, data_root, load_universe, save_year, validate_bars

SOURCE_COLUMNS = ('symbol','isin','year','yahoo_rows','kaggle_rows','overlap_dates',
    'kaggle_only_dates','kaggle_metadata_isin','isin_mismatch','first_yahoo_date',
    'first_kaggle_date','verified_listing_date','yahoo_prelisting_rows',
    'kaggle_prelisting_rows','median_kaggle_to_yahoo_ratio',
    'max_abs_ratio_step_pct','yahoo_large_jumps','source_sha256','finding',
    'decision','approved_source','trim_prelisting','evidence_url', 'review_note')
DECISIONS = {'requires_review','approved_adjusted','must_exclude'}
CORPORATE_ACTION_RULES = (
    ('TCS','2006-08-09','bonus','2','https://www.tcs.com/investor-relations/investor-faqs'),
    ('TCS','2009-06-18','bonus','2','https://www.tcs.com/investor-relations/investor-faqs'),
    ('TCS','2018-06-04','bonus','2','https://www.tcs.com/investor-relations/investor-faqs'),
)

LISTING_RULES = (
    ('TCS','2004-08-25','https://nsearchives.nseindia.com/content/press/23082004.pdf'),
    ('BAJAJ-AUTO','2008-05-26','https://nsearchives.nseindia.com/content/press/22052008.htm'),
    ('BAJAJFINSV','2008-05-26','https://nsearchives.nseindia.com/content/press/22052008.htm'),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(item,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n',encoding='utf-8')
    tmp.replace(path)


def price_provider(version: str) -> str:
    if not isinstance(version,str) or not re.fullmatch(r'v[a-z0-9][a-z0-9_-]{0,30}',version):
        raise ValueError('Version must start with v and contain lowercase letters/digits/_/- (e.g. v2)')
    return 'research_'+version


def _report_path(root,universe_name,snapshot,version):
    return Path(root)/'reports'/'historical_validation'/f'{universe_name}_{snapshot}_{version}_review.csv'


def _listing_rules(root):
    path=Path(root)/'reports'/'historical_validation'/'listing_rules.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        pd.DataFrame(LISTING_RULES,columns=['symbol','listed_from','evidence_url']).to_csv(path,index=False)
    rules=pd.read_csv(path,dtype=str).fillna('')
    if not {'symbol','listed_from','evidence_url'}.issubset(rules) or rules.symbol.duplicated().any():
        raise ValueError('Listing rules need unique symbol,listed_from,evidence_url')
    out={}
    for rec in rules.itertuples(index=False):
        if rec.listed_from:
            listed=pd.Timestamp(rec.listed_from).normalize()
            if listed.year < 1990 or not rec.evidence_url.strip():
                raise ValueError(f'Missing evidence/invalid listing rule for {rec.symbol}')
            out[rec.symbol]=(listed, rec.evidence_url)
    return out,path


def _load_prices(path:Path,symbol,isin,year):
    if not path.is_file(): return None
    data=validate_bars(pd.read_csv(path,dtype={'symbol':str,'isin':str}),symbol=symbol)
    if data['isin'].iloc[0]!=isin or not data.date.dt.year.eq(year).all():
        raise ValueError(f'Stock/year/ISIN mismatch: {path}')
    return data


def _kaggle_isins(root):
    p=Path(root)/'kaggle'/'stock_metadata.csv'
    if not p.is_file():return {}
    d=pd.read_csv(p,dtype=str).fillna('')
    if not {'Symbol','ISIN Code'}.issubset(d):
        raise ValueError('Invalid Kaggle stock_metadata.csv')
    if d.Symbol.duplicated().any():raise ValueError('Duplicate Kaggle symbols')
    return dict(zip(d.Symbol.str.upper(),d['ISIN Code'].str.upper()))


def _corporate_actions(root):
    path=Path(root)/'reports'/'historical_validation'/'corporate_actions.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        pd.DataFrame(CORPORATE_ACTION_RULES,columns=[
            'symbol','effective_date','event_type','share_factor','evidence_url']).to_csv(path,index=False)
    frame=pd.read_csv(path,dtype=str).fillna('')
    if not {'symbol','effective_date','event_type','share_factor','evidence_url'}.issubset(frame):
        raise ValueError('corporate_actions.csv needs symbol,effective_date,event_type,share_factor,evidence_url')
    output={}
    for r in frame.itertuples(index=False):
        event=pd.Timestamp(r.effective_date).normalize()
        if not r.evidence_url.strip() or not r.share_factor or float(r.share_factor)<=0:
            raise ValueError(f'Invalid corporate-action evidence or factor for {r.symbol}')
        output.setdefault(r.symbol,[]).append((event,r.event_type,float(r.share_factor),r.evidence_url))
    return output,path


def audit(*,root=None,snapshot,version='v2',universe_name='nifty50',
          start='2000-01-01',end='2019-12-31'):
    """Generate/editable review manifest. Preserve operator's decisions on repeated audit."""
    root=data_root(root); provider=price_provider(version)
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if (start.month,start.day)!=(1,1) or (end.month,end.day)!=(12,31) or start>end:
        raise ValueError('Audit uses full inclusive calendar years')
    universe=load_universe(root,snapshot,universe_name)
    rules,rules_path=_listing_rules(root)
    kaggle_isins=_kaggle_isins(root)
    corporate,corporate_path=_corporate_actions(root)
    dest=_report_path(root,universe_name,snapshot,version)
    old={}
    if dest.is_file():
        prior=pd.read_csv(dest,dtype=str,keep_default_na=False)
        if not set(SOURCE_COLUMNS).issubset(prior):
            raise ValueError('Invalid existing review CSV; do not overwrite manual decisions')
        if prior.duplicated(['isin','year']).any():raise ValueError('Duplicate review rows')
        old={(r['isin'],int(r['year'])):r for r in prior.to_dict('records')}
    rows=[];action_checks=[]
    for stock in universe.itertuples(index=False):
        listed,rule_source=rules.get(stock.symbol,(pd.NaT,''))
        k_isin=kaggle_isins.get(stock.symbol,'')
        mismatch=bool(k_isin and k_isin != stock.isin)
        for year in range(start.year,end.year+1):
            yahoo_path=bar_path(root,stock.isin,year,'yahoo_finance')
            kaggle_path=bar_path(root,stock.isin,year,'kaggle_raw')
            finding=[]; yahoo=None;kaggle=None; failure=None
            try:
                yahoo=_load_prices(yahoo_path,stock.symbol,stock.isin,year)
                kaggle=_load_prices(kaggle_path,stock.symbol,stock.isin,year)
            except Exception as exc:
                failure=f'{type(exc).__name__}: {exc}'
                finding.append('invalid_source:'+failure[:140])
            if mismatch: finding.append('kaggle_isin_mismatch_requires_corporate_action_check')
            if yahoo is None: finding.append('no_yahoo_adjusted_source')
            if kaggle is not None and yahoo is None: finding.append('kaggle_raw_only_do_not_promote')
            y_pre=int((yahoo.date<listed).sum()) if yahoo is not None and pd.notna(listed) else 0
            k_pre=int((kaggle.date<listed).sum()) if kaggle is not None and pd.notna(listed) else 0
            if y_pre:finding.append('yahoo_dates_before_verified_listing')
            if k_pre:finding.append('kaggle_dates_before_verified_listing')
            overlap=0; only=0; median=None; ratio_step=None
            if kaggle is not None:
                only=len(kaggle) if yahoo is None else int((~kaggle.date.isin(yahoo.date)).sum())
            joined=None
            if yahoo is not None and kaggle is not None:
                joined=yahoo[['date','close']].merge(kaggle[['date','close']],on='date',suffixes=('_yahoo','_kaggle')).sort_values('date')
                overlap=len(joined)
                if overlap:
                    ratio=joined.close_kaggle/joined.close_yahoo
                    median=float(ratio.median())
                    ratio_step=float((ratio/ratio.shift(1)-1).abs().max()*100) if overlap>1 else 0.0
                    if ratio_step>20:finding.append('cross_source_adjustment_factor_jump_gt_20pct')
            jumps=0
            if yahoo is not None:
                jumps=int(yahoo.close.pct_change(fill_method=None).abs().gt(.4).sum())
                if jumps:finding.append('yahoo_close_jump_gt_40pct')
            for event,event_type,share_factor,evidence in corporate.get(stock.symbol,[]):
                if event.year != year:continue
                finding.append('documented_corporate_action_requires_adjustment_review')
                before=None;after=None;observed=None
                if joined is not None and not joined.empty:
                    preceding=joined.loc[joined.date<event].tail(1)
                    following=joined.loc[joined.date>=event].head(1)
                    if not preceding.empty and not following.empty:
                        before=float(preceding.close_kaggle.iloc[0]/preceding.close_yahoo.iloc[0])
                        after=float(following.close_kaggle.iloc[0]/following.close_yahoo.iloc[0])
                        observed=after/before if before else None
                action_checks.append({'symbol':stock.symbol,'year':year,'effective_date':event.date(),
                    'event_type':event_type,'expected_share_factor':share_factor,'evidence_url':evidence,
                    'ratio_before':before,'ratio_after':after,'observed_ratio_change':observed,
                    'requires_manual_review':True})
            if pd.notna(listed) and (listed.year>year):
                finding.append('entire_year_before_verified_listing')
                status='must_exclude'
            else:
                status='requires_review'
            if not finding: finding=['yahoo_adjustment_and_security_identity_need_independent_review']
            previous=old.get((stock.isin,year))
            retain=bool(previous and
                        previous.get('source_sha256','')==(_digest(yahoo_path) if yahoo_path.is_file() else '') and
                        previous.get('kaggle_metadata_isin','')==k_isin and
                        previous.get('verified_listing_date','')==(str(listed.date()) if pd.notna(listed) else ''))
            rec={'symbol':stock.symbol,'isin':stock.isin,'year':year,
                 'yahoo_rows':len(yahoo) if yahoo is not None else 0,
                 'kaggle_rows':len(kaggle) if kaggle is not None else 0,
                 'overlap_dates':overlap,'kaggle_only_dates':only,
                 'kaggle_metadata_isin':k_isin,'isin_mismatch':str(mismatch).lower(),
                 'first_yahoo_date':str(yahoo.date.min().date()) if yahoo is not None else '',
                 'first_kaggle_date':str(kaggle.date.min().date()) if kaggle is not None else '',
                 'verified_listing_date':str(listed.date()) if pd.notna(listed) else '',
                 'yahoo_prelisting_rows':y_pre,'kaggle_prelisting_rows':k_pre,
                 'median_kaggle_to_yahoo_ratio':median,'max_abs_ratio_step_pct':ratio_step,
                 'yahoo_large_jumps':jumps,
                 'source_sha256':_digest(yahoo_path) if yahoo_path.is_file() else '',
                 'finding':';'.join(finding),
                 'decision':previous['decision'] if retain else status,
                 'approved_source':previous['approved_source'] if retain else 'yahoo_finance',
                 'trim_prelisting':previous['trim_prelisting'] if retain else 'false',
                 'evidence_url':previous['evidence_url'] if retain else (rule_source if status=='must_exclude' else ''),
                 'review_note':previous['review_note'] if retain else ''}
            rows.append(rec)
    result=pd.DataFrame(rows,columns=SOURCE_COLUMNS)
    tmp=dest.with_suffix('.csv.tmp');tmp.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(tmp,index=False);tmp.replace(dest)
    actions_path=dest.with_name(dest.stem+'_corporate_actions.csv')
    pd.DataFrame(action_checks,columns=['symbol','year','effective_date','event_type',
        'expected_share_factor','evidence_url','ratio_before','ratio_after',
        'observed_ratio_change','requires_manual_review']).to_csv(actions_path,index=False)
    report={'version':version,'price_source':provider,'universe_name':universe_name,
        'snapshot':snapshot,'start':start.strftime('%Y-%m-%d'),'end':end.strftime('%Y-%m-%d'),
        'rows':len(result),'requires_review':int((result.decision=='requires_review').sum()),
        'must_exclude':int((result.decision=='must_exclude').sum()),
        'approved_adjusted':int((result.decision=='approved_adjusted').sum()),
        'yahoo_prelisting_stock_years':int((result.yahoo_prelisting_rows>0).sum()),
        'kaggle_isin_mismatches':int(result.loc[result.isin_mismatch=='true','symbol'].nunique()),
        'review_file':str(dest),'listing_rules_file':str(rules_path),
        'corporate_action_rules_file':str(corporate_path),
        'corporate_action_comparisons_file':str(actions_path),'sources_modified':False}
    _write_json(dest.with_suffix('.json'),report)
    return report


def _truth(v):
    if str(v).strip().lower() not in ('true','false'):
        raise ValueError(f'trim_prelisting must be true/false; got {v}')
    return str(v).strip().lower()=='true'


def build(*,root=None,snapshot,version='v2',universe_name='nifty50',
          start='2000-01-01',end='2016-12-31'):
    """Publish reviewed, non-overwriting stock-year files for a versioned model run.

    All stock-years must be explicitly classified; data are never promoted from raw
    Kaggle. A second build (holdout years) is needed before model training.
    """
    root=data_root(root);provider=price_provider(version)
    first,last=pd.Timestamp(start),pd.Timestamp(end)
    if first>last or (first.month,first.day)!=(1,1) or (last.month,last.day)!=(12,31):
        raise ValueError('Build uses full calendar years')
    universe=load_universe(root,snapshot,universe_name)
    manifest=_report_path(root,universe_name,snapshot,version)
    if not manifest.is_file():raise FileNotFoundError(f'Run audit first: {manifest}')
    review=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    if not set(SOURCE_COLUMNS).issubset(review):raise ValueError('Invalid review CSV')
    if review.duplicated(['isin','year']).any():raise ValueError('Duplicate review row')
    portion=review[review.year.astype(int).between(first.year,last.year)].copy()
    expected=len(universe)*(last.year-first.year+1)
    if len(portion)!=expected:raise ValueError(f'Incomplete review manifest: {len(portion)} vs {expected}')
    members=set(zip(universe['isin'],universe['symbol']))
    if any((r.isin,r.symbol) not in members for r in portion.itertuples()):
        raise ValueError('Review contains an unknown/changed security')
    rules,_=_listing_rules(root)
    to_publish=[];skipped=[];pending=[];undocumented=[]
    for r in portion.itertuples(index=False):
        key=f'{r.symbol}/{r.year}'
        if r.decision not in DECISIONS: raise ValueError(f'Invalid decision {r.decision} for {key}')
        if r.decision=='requires_review': pending.append(key);continue
        if r.decision=='must_exclude': skipped.append(key);continue
        # Review annotations are optional; record their absence rather than failing.
        if not (r.evidence_url.strip() and r.review_note.strip()):
            undocumented.append(key)
        if r.approved_source=='kaggle_raw' or r.approved_source=='nse_raw' or r.approved_source=='research_ohlcv':
            raise ValueError(f'Unadjusted/unverified source forbidden for {key}')
        if r.approved_source!='yahoo_finance' and not re.fullmatch(r'ext_[a-z0-9_-]+',r.approved_source):
            raise ValueError(f'Only Yahoo or independently adjusted ext_ source allowed: {key}')
        source=bar_path(root,r.isin,int(r.year),r.approved_source)
        if not source.is_file():raise FileNotFoundError(f'Approved {key}: missing adjusted source {source}')
        data=_load_prices(source,r.symbol,r.isin,int(r.year))
        if r.approved_source=='yahoo_finance' and _digest(source)!=r.source_sha256:
            raise ValueError(f'Yahoo source changed since audit: {key}; rerun audit with a new version')
        if r.approved_source.startswith('ext_'):
            extmeta=source.with_name(f'{r.year}.source.json')
            if not extmeta.is_file() or json.loads(extmeta.read_text()).get('cache_sha256')!=_digest(source):
                raise ValueError(f'Approved external source needs verified import metadata: {key}')
        listed,rule_url=rules.get(r.symbol,(pd.NaT,''))
        trimmed=0
        if pd.notna(listed):
            trimmed=int((data.date<listed).sum())
            if trimmed:
                if not _truth(r.trim_prelisting):
                    raise ValueError(f'{key} has {trimmed} rows before verified listing; mark trim_prelisting=true after review')
                data=data[data.date>=listed].copy()
                if data.empty:raise ValueError(f'{key}: all data prelisting; mark must_exclude instead')
        to_publish.append((r,source,data,trimmed))
    if pending:
        raise ValueError(f'Cannot build: {len(pending)} unreviewed stock-years (e.g. {pending[:8]}); classify approved_adjusted or must_exclude in {manifest}')
    out_dir=root/'reports'/provider
    summary_path=out_dir/f'{universe_name}_{snapshot}_{start}_{end}_summary.json'
    if summary_path.is_file():
        earlier=json.loads(summary_path.read_text())
        if earlier.get('manifest_sha256') != _digest(manifest):
            raise ValueError('Dataset version already published with another review manifest; choose a new version')
    # Perform complete preflight, including existing data immutability, BEFORE any writes.
    for r,source,data,trimmed in to_publish:
        target=bar_path(root,r.isin,int(r.year),provider)
        source_meta=target.with_name(f'{r.year}.source.json')
        if target.exists():
            if not source_meta.exists():raise ValueError(f'Missing version provenance: {target}')
            old=json.loads(source_meta.read_text())
            if (old.get('source_sha256')!=_digest(source) or
                old.get('source')!=r.approved_source or
                old.get('trimmed_prelisting_rows')!=trimmed or
                old.get('canonical_sha256')!=_digest(target)):
                raise ValueError(f'Existing dataset {version} is immutable; create a NEW version instead of changing {target}')
            old_frame=_load_prices(target,r.symbol,r.isin,int(r.year))
            if not old_frame.reset_index(drop=True).equals(validate_bars(data).reset_index(drop=True)):
                raise ValueError(f'Existing approved candles differ in version {version}: {target}')
    selected=0
    for r,source,data,trimmed in to_publish:
        target=save_year(root,data,isin=r.isin,year=int(r.year),provider=provider)
        meta_path=target.with_name(f'{r.year}.source.json')
        newmeta={'version':version,'source':r.approved_source,
                 'source_sha256':_digest(source),'canonical_sha256':_digest(target),
                 'symbol':r.symbol,'isin':r.isin,'year':int(r.year),
                 'listing_date':r.verified_listing_date,
                 'trimmed_prelisting_rows':trimmed,'evidence_url':r.evidence_url,
                 'review_note':r.review_note,
                 'approval_documented':bool(r.evidence_url.strip() and r.review_note.strip()),
                 'price_adjustment_reviewed_by_user':True,
                 'live_trading_enabled':False}
        if not meta_path.exists():_write_json(meta_path,newmeta)
        selected+=1
    out_dir=root/'reports'/provider;out_dir.mkdir(parents=True,exist_ok=True)
    summary={'provider':provider,'version':version,'snapshot':snapshot,
             'universe_name':universe_name,'universe_size':len(universe),
             'start':first.strftime('%Y-%m-%d'),'end':last.strftime('%Y-%m-%d'),
             'requests_or_cache_checks':expected,'selected':selected,
             'excluded':len(skipped),'pending':0,'errors':0,
             'approvals_missing_evidence_or_note':len(undocumented),
             'warning':('Some approved stock-years lack evidence_url or review_note; '
                        'approvals are manual and their price adjustments are not independently verified.'
                        if undocumented else ''),
             'manifest_sha256':_digest(manifest),'audited_review_manifest':str(manifest),
             'live_execution_enabled':False}
    name=f'{universe_name}_{snapshot}_{start}_{end}_summary.json'
    _write_json(out_dir/name,summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root')
    p.add_argument('--snapshot',required=True)
    p.add_argument('--universe-name',default='nifty50')
    p.add_argument('--version',default='v2')
    action=p.add_subparsers(dest='action',required=True)
    scan=action.add_parser('audit',help='Write review CSV; never modify source data')
    scan.add_argument('--from-date',default='2000-01-01')
    scan.add_argument('--to-date',default='2019-12-31')
    publish=action.add_parser('build',help='Create versioned dataset from manually approved adjustments only')
    publish.add_argument('--from-date',required=True)
    publish.add_argument('--to-date',required=True)
    args=p.parse_args()
    if args.action=='audit':
        result=audit(root=args.data_root,snapshot=args.snapshot,universe_name=args.universe_name,
                     version=args.version,start=args.from_date,end=args.to_date)
    else:
        result=build(root=args.data_root,snapshot=args.snapshot,universe_name=args.universe_name,
                     version=args.version,start=args.from_date,end=args.to_date)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
