"""Focused offline tests for the CSV review-to-repair workflow."""
import hashlib
import json
import pandas as pd
import pytest
from marketdata.shared_store import save_year, bar_path
from marketdata.price_issue_review import scan, apply

SNAP='2026-10-18'
NAME='review_demo'

def make_data(root):
    universe=root/'universes'/NAME
    universe.mkdir(parents=True)
    pd.DataFrame({'symbol':['TEST','OTHER'], 'isin':['INE123A01010','INE123B01010'],
                  'yahoo_ticker':['TEST.NS','OTHER.NS']}).to_csv(universe/(SNAP+'.csv'),index=False)
    for symbol,isin in [('TEST','INE123A01010'),('OTHER','INE123B01010')]:
        for year in (2016,2017):
            bars=pd.DataFrame({'date':pd.to_datetime([f'{year}-01-02',f'{year}-01-03',f'{year}-01-04']),
                'symbol':symbol,'isin':isin,'open':[100.,150.,100.],
                'high':[110.,160.,110.],'low':[90.,140.,90.],
                'close':[100.,150.,100.],'volume':[1000.,2000.,3000.]})
            path=save_year(root,bars,isin=isin,year=year,provider='research_v3')
            path.with_name(f'{year}.source.json').write_text(json.dumps({
                'canonical_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'symbol':symbol,'isin':isin,'year':year,
                'source':'kaggle_raw' if year==2016 else 'yahoo_finance',
                'price_basis':'raw_unadjusted' if year==2016 else 'provider_adjusted_unverified',
                'user_approved_raw':year==2016,
                'price_adjustment_reviewed_by_user':year!=2016}))
    for a,b in [('2000-01-01','2016-12-31'),('2017-01-01','2019-12-31')]:
        report=root/'reports'/'research_v3'/f'{NAME}_{SNAP}_{a}_{b}_summary.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps({'provider':'research_v3','start':a,'end':b,
            'snapshot':SNAP,'universe_name':NAME,'universe_size':2,
            'requests_or_cache_checks':34 if a.startswith('2000') else 6,
            'errors':0,'pending':0,'manifest_sha256':'abc'}))
    return root

def issue_csv(root):
    summary=scan(root=root,provider='research_v3',snapshot=SNAP,universe_name=NAME,
                 start='2016-01-01',end='2017-12-31')
    path=summary['review_file']
    return path,pd.read_csv(path,dtype=str,keep_default_na=False)

def test_reviewed_repair_is_new_version_and_immutable(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    match=rows.index[rows.symbol.eq('TEST') & rows.date.eq('2016-01-03')]
    assert len(match)==1
    idx=match[0]
    old=bar_path(root,'INE123A01010',2016,'research_v3')
    before=old.read_bytes()
    rows.loc[idx,'verification_result']='verified_issue'
    rows.loc[idx,'correction_action']='replace_bar'
    rows.loc[idx,'verification_notes']='Checked NSE report'
    rows.loc[idx,'reference_url']='https://example.org/document'
    for col,val in dict(open='100',high='110',low='90',close='100',volume='2000').items():
        rows.loc[idx,'corrected_'+col]=val
    rows.to_csv(path,index=False)
    arguments=dict(root=root,parent='research_v3',version='research_v4',
        snapshot=SNAP,universe_name=NAME,review_file=path)
    assert apply(**arguments,dry_run=True)['replaced_bars']==1
    assert apply(**arguments)['replaced_bars']==1
    assert old.read_bytes()==before
    new=bar_path(root,'INE123A01010',2016,'research_v4')
    assert pd.read_csv(new).loc[lambda x:x.date.eq('2016-01-03'),'close'].iloc[0]==100
    metadata=json.loads(new.with_name('2016.source.json').read_text())
    assert metadata['source']=='kaggle_raw' and len(metadata['verified_bar_corrections'])==1
    with pytest.raises(FileExistsError):
        apply(**arguments)

def test_unreviewed_correction_rejected(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    idx=rows.index[rows.symbol.eq('TEST') & rows.date.eq('2016-01-03')][0]
    rows.loc[idx,'correction_action']='replace_bar'
    rows.to_csv(path,index=False)
    with pytest.raises(ValueError,match='verified_issue'):
        apply(root=root,parent='research_v3',version='research_v4',
              snapshot=SNAP,universe_name=NAME,review_file=path,dry_run=True)

def test_stock_year_exclusion(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    idx=rows.index[rows.symbol.eq('TEST') & rows.date.eq('2016-01-03')][0]
    rows.loc[idx,'correction_action']='exclude_stock_year'
    rows.loc[idx,'verification_result']='verified_issue'
    rows.loc[idx,'verification_notes']='Misattributed data confirmed'
    rows.loc[idx,'reference_url']='https://example.org/evidence'
    rows.to_csv(path,index=False)
    summary=apply(root=root,parent='research_v3',version='research_v4',
        snapshot=SNAP,universe_name=NAME,review_file=path)
    assert summary['excluded_stock_years']==1
    assert not bar_path(root,'INE123A01010',2016,'research_v4').exists()


def test_verified_bar_deletion_keeps_other_dates_and_source(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    idx=rows.index[rows.symbol.eq('TEST') & rows.date.eq('2016-01-03')][0]
    rows.loc[idx,'verification_result']='verified_issue'
    rows.loc[idx,'verification_notes']='Dated exchange notice confirms no equity trading on this session'
    rows.loc[idx,'reference_url']='https://example.org/archive/holiday'
    rows.loc[idx,'correction_action']='delete_bar'
    rows.to_csv(path,index=False)
    old=bar_path(root,'INE123A01010',2016,'research_v3')
    original=old.read_bytes()
    args=dict(root=root,parent='research_v3',version='research_v4',
              snapshot=SNAP,universe_name=NAME,review_file=path)
    preview=apply(**args,dry_run=True)
    assert preview['deleted_bars']==1
    assert preview['replaced_bars']==0
    result=apply(**args)
    assert result['deleted_bars']==1
    assert old.read_bytes()==original
    target=bar_path(root,'INE123A01010',2016,'research_v4')
    candles=pd.read_csv(target)
    assert len(candles)==2
    assert '2016-01-03' not in candles.date.tolist()
    meta=json.loads(target.with_name('2016.source.json').read_text())
    assert meta['verified_bar_deletions'][0]['day']=='2016-01-03'
    summary=root/'reports'/'research_v4'/f'{NAME}_{SNAP}_2000-01-01_2016-12-31_summary.json'
    assert json.loads(summary.read_text())['verified_deleted_bars']==1
    with pytest.raises(FileExistsError):
        apply(**args)


def test_unverified_bar_deletion_rejected(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    idx=rows.index[rows.symbol.eq('TEST') & rows.date.eq('2016-01-03')][0]
    rows.loc[idx,'correction_action']='delete_bar'
    rows.to_csv(path,index=False)
    with pytest.raises(ValueError,match='verified_issue'):
        apply(root=root,parent='research_v3',version='research_v4',
              snapshot=SNAP,universe_name=NAME,review_file=path,dry_run=True)


def test_deleting_all_rows_in_partition_requires_stock_year_exclusion(tmp_path):
    root=make_data(tmp_path/'data')
    path,rows=issue_csv(root)
    matches=rows.index[rows.symbol.eq('TEST') & rows.year.eq('2016')]
    # Scan flags Jan 3 and Jan 4. Deleting both still leaves Jan 2.
    assert len(matches)==2
    for idx in matches:
        rows.loc[idx,'verification_result']='verified_issue'
        rows.loc[idx,'verification_notes']='External trading-calendar validation'
        rows.loc[idx,'reference_url']='https://example.org/market-closed'
        rows.loc[idx,'correction_action']='delete_bar'
    rows.to_csv(path,index=False)
    result=apply(root=root,parent='research_v3',version='research_v4',
                 snapshot=SNAP,universe_name=NAME,review_file=path)
    assert result['deleted_bars']==2
    kept=pd.read_csv(bar_path(root,'INE123A01010',2016,'research_v4'))
    assert kept.date.tolist()==['2016-01-02']
