"""Offline regression tests for the security-history and approved-data quarantine."""
import json

import numpy as np
import pandas as pd
import pytest

from marketdata.historical_validation import audit, build, price_provider
from marketdata.shared_store import save_year,bar_path
from models.nifty50_stage1 import verify_download_audits, verify_versioned_provenance

SNAP='2026-10-18'
NAME='review_test'


def universe(root):
    folder=root/'universes'/NAME;folder.mkdir(parents=True)
    frame=pd.DataFrame({
        'symbol':['TCS','BAJAJ-AUTO','ITC'],
        'isin':['INE467B01029','INE917I01010','INE154A01025'],
        'yahoo_ticker':['TCS.NS','BAJAJ-AUTO.NS','ITC.NS']})
    frame.to_csv(folder/f'{SNAP}.csv',index=False)
    return frame


def prices(symbol,isin,year=2004,days=None):
    dates=pd.DatetimeIndex(days or [f'{year}-06-01',f'{year}-09-01',f'{year}-09-02'])
    n=len(dates)
    return pd.DataFrame({'date':dates,'symbol':symbol,'isin':isin,
                         'open':np.full(n,100.),'high':np.full(n,102.),
                         'low':np.full(n,99.),'close':np.full(n,101.),
                         'volume':np.full(n,10000.)})


def setup(tmp_path):
    root=tmp_path/'data';frame=universe(root)
    for symbol,isin in zip(frame.symbol,frame['isin']):
        save_year(root,prices(symbol,isin),isin=isin,year=2004)
    return root


def manifest(root, version='v2'):
    return root/'reports'/'historical_validation'/f'{NAME}_{SNAP}_{version}_review.csv'


def test_listing_gates_and_identity_mismatch_report(tmp_path):
    root=setup(tmp_path)
    kaggle=root/'kaggle';kaggle.mkdir()
    pd.DataFrame({'Symbol':['TCS','BAJAJ-AUTO','ITC'],
                  'ISIN Code':['INE467B01020','INE917I01010','INE154A01025']}).to_csv(kaggle/'stock_metadata.csv',index=False)
    r=audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    d=pd.read_csv(r['review_file']).set_index('symbol')
    assert r['yahoo_prelisting_stock_years']==2
    assert r['kaggle_isin_mismatches']==1
    assert d.loc['BAJAJ-AUTO','decision']=='must_exclude'
    assert d.loc['TCS','decision']=='requires_review'
    assert d.loc['TCS','yahoo_prelisting_rows']==1
    assert 'kaggle_isin_mismatch' in d.loc['TCS','finding']
    assert bar_path(root,'INE467B01029',2004,'yahoo_finance').exists()
    assert not bar_path(root,'INE467B01029',2004,price_provider('v2')).exists()


def test_edit_preservation_and_prelisting_trim(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);data=pd.read_csv(p,dtype=str).fillna('')
    for stock in ['TCS','ITC']:
        data.loc[data.symbol==stock,'decision']='approved_adjusted'
        data.loc[data.symbol==stock,'review_note']='Checked source, listing identity and adjustments'
        data.loc[data.symbol==stock,'evidence_url']='https://nsearchives.nseindia.com/content/press/23082004.pdf'
    data.loc[data.symbol=='TCS','trim_prelisting']='true'
    data.to_csv(p,index=False)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert pd.read_csv(p).loc[lambda df:df.symbol.eq('TCS'),'decision'].iloc[0]=='approved_adjusted'
    published=build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert published['selected']==2 and published['excluded']==1
    dest=bar_path(root,'INE467B01029',2004,'research_v2')
    saved=pd.read_csv(dest)
    assert len(saved)==2 and pd.to_datetime(saved.date).min()==pd.Timestamp('2004-09-01')
    meta=json.loads(dest.with_name('2004.source.json').read_text())
    assert meta['trimmed_prelisting_rows']==1
    assert json.loads((root/'reports'/'research_v2'/f'{NAME}_{SNAP}_2004-01-01_2004-12-31_summary.json').read_text())['selected']==2
    assert build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')['selected']==2
    edited=pd.read_csv(p,dtype=str).fillna('');edited.loc[edited.symbol=='ITC','review_note']='changed approval';edited.to_csv(p,index=False)
    with pytest.raises(ValueError,match='another review manifest'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')


def test_unreviewed_and_prelisting_guard_fail_closed(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    with pytest.raises(ValueError,match='unreviewed'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str).fillna('')
    d.loc[d.symbol=='TCS','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.decision=='approved_adjusted','review_note']='reviewed adjustments'
    d.loc[d.decision=='approved_adjusted','evidence_url']='https://example.org/evidence'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='trim_prelisting=true'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert not bar_path(root,'INE154A01025',2004,'research_v2').exists()


def test_raw_kaggle_never_accepted_and_external_provenance(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str).fillna('')
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','evidence_url']='https://example.org/reviewed'
    d.loc[d.symbol=='ITC','review_note']='Adjusted CSV verified'
    d.loc[d.symbol=='ITC','approved_source']='nse_raw'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Unreviewed source forbidden'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    d.loc[d.symbol=='ITC','approved_source']='ext_checked'
    d.to_csv(p,index=False)
    with pytest.raises(FileNotFoundError,match='missing selected source'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')


def test_dynamic_model_audits_verified_version_and_mismatch(tmp_path):
    root=tmp_path/'data';universe(root)
    folder=root/'reports'/'research_v2';folder.mkdir(parents=True)
    for start,end,count in [('2000-01-01','2016-12-31',17),('2017-01-01','2019-12-31',3)]:
        (folder/f'{NAME}_{SNAP}_{start}_{end}_summary.json').write_text(json.dumps({
            'provider':'research_v2','universe_name':NAME,'snapshot':SNAP,
            'start':start,'end':end,'universe_size':3,'requests_or_cache_checks':3*count,
            'errors':0,'pending':0,'manifest_sha256':'abc'}))
    assert len(verify_download_audits(root,SNAP,NAME,'research_v2'))==2
    f=folder/f'{NAME}_{SNAP}_2017-01-01_2019-12-31_summary.json'
    wrong=json.loads(f.read_text());wrong['manifest_sha256']='other';f.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='Different review manifest'):
        verify_download_audits(root,SNAP,NAME,'research_v2')


@pytest.mark.parametrize('evidence,note', [('', ''), ('https://example.org/evidence', ''), ('', 'Reviewed manually')])
def test_approval_without_evidence_or_note_is_allowed_but_flagged(tmp_path, evidence, note):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','evidence_url']=evidence
    d.loc[d.symbol=='ITC','review_note']=note
    d.to_csv(p,index=False)
    result=build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert result['selected']==1
    assert result['excluded']==2
    assert result['approvals_missing_evidence_or_note']==1
    assert 'not independently verified' in result['warning']
    target=bar_path(root,'INE154A01025',2004,'research_v2')
    assert target.is_file()
    metadata=json.loads(target.with_name('2004.source.json').read_text())
    assert metadata['approval_documented'] is False
    assert metadata['evidence_url']==evidence
    assert metadata['review_note']==note


def test_complete_approval_has_no_documentation_warning(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol!='ITC','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','evidence_url']='https://example.org/evidence'
    d.loc[d.symbol=='ITC','review_note']='Reviewed'
    d.to_csv(p,index=False)
    result=build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert result['approvals_missing_evidence_or_note']==0
    assert result['warning']==''
    target=bar_path(root,'INE154A01025',2004,'research_v2')
    metadata=json.loads(target.with_name('2004.source.json').read_text())
    assert metadata['approval_documented'] is True


def test_explicit_kaggle_raw_build_has_unadjusted_provenance(tmp_path):
    root=setup(tmp_path)
    kaggle=root/'kaggle';kaggle.mkdir()
    pd.DataFrame({'Symbol':['TCS','BAJAJ-AUTO','ITC'],
                  'ISIN Code':['INE467B01029','INE917I01010','INE154A01025']}).to_csv(kaggle/'stock_metadata.csv',index=False)
    save_year(root, prices('ITC','INE154A01025',2004), isin='INE154A01025', year=2004, provider='kaggle_raw')
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','approved_source']='kaggle_raw'
    d.to_csv(p,index=False)
    result=build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert result['selected']==1
    assert result['raw_kaggle_stock_years']==1
    assert 'not split/dividend adjusted' in result['warning'].lower()
    target=bar_path(root,'INE154A01025',2004,'research_v2')
    assert target.exists()
    metadata=json.loads(target.with_name('2004.source.json').read_text())
    assert metadata['source']=='kaggle_raw'
    assert metadata['price_basis']=='raw_unadjusted'
    assert metadata['user_approved_raw'] is True
    assert metadata['price_adjustment_reviewed_by_user'] is False
    assert bar_path(root,'INE154A01025',2004,'yahoo_finance').exists()
    assert build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')['raw_kaggle_stock_years']==1


def test_kaggle_isin_mismatch_does_not_relabel_security(tmp_path):
    root=setup(tmp_path)
    kaggle=root/'kaggle';kaggle.mkdir()
    pd.DataFrame({'Symbol':['TCS','ITC'], 'ISIN Code':['INE467B01029','INE154A01017']}).to_csv(kaggle/'stock_metadata.csv',index=False)
    save_year(root, prices('ITC','INE154A01025',2004), isin='INE154A01025', year=2004, provider='kaggle_raw')
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','approved_source']='kaggle_raw'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Kaggle ISIN/metadata mismatch'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    assert not bar_path(root,'INE154A01025',2004,'research_v2').exists()


def test_kaggle_selected_without_cached_partition_errors_clearly(tmp_path):
    root=setup(tmp_path)
    kaggle=root/'kaggle';kaggle.mkdir()
    pd.DataFrame({'Symbol':['ITC'], 'ISIN Code':['INE154A01025']}).to_csv(kaggle/'stock_metadata.csv',index=False)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','approved_source']='kaggle_raw'
    d.to_csv(p,index=False)
    with pytest.raises(FileNotFoundError,match='run Kaggle audit'):
        build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')


def test_model_fingerprint_accepts_only_explicit_raw_provenance(tmp_path):
    root=setup(tmp_path)
    kaggle=root/'kaggle';kaggle.mkdir()
    pd.DataFrame({'Symbol':['ITC'], 'ISIN Code':['INE154A01025']}).to_csv(kaggle/'stock_metadata.csv',index=False)
    save_year(root,prices('ITC','INE154A01025',2004),isin='INE154A01025',year=2004,provider='kaggle_raw')
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','approved_source']='kaggle_raw'
    d.to_csv(p,index=False)
    build(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    symbols=pd.read_csv(root/'universes'/NAME/f'{SNAP}.csv',dtype=str)
    fingerprint,count=verify_versioned_provenance(root,symbols,'research_v2','manifest-sha')
    assert len(fingerprint)==64
    assert count==1
    target=bar_path(root,'INE154A01025',2004,'research_v2')
    meta=target.with_name('2004.source.json')
    data=json.loads(meta.read_text())
    data['user_approved_raw']=False
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='explicit unadjusted approval'):
        verify_versioned_provenance(root,symbols,'research_v2','manifest-sha')
    data['user_approved_raw']=True
    meta.write_text(json.dumps(data))
    target.write_bytes(b'corrupt file')
    with pytest.raises(ValueError,match='changed version partition'):
        verify_versioned_provenance(root,symbols,'research_v2','manifest-sha')


def test_approved_entirely_prelisting_year_excluded_without_editing_review(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    # BAJAJ-AUTO's independently verified listing is 2008, not 2004.
    d.loc[d.symbol=='BAJAJ-AUTO','decision']='approved_adjusted'
    d.loc[d.symbol=='BAJAJ-AUTO','approved_source']='kaggle_raw'
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.to_csv(p,index=False)
    original=p.read_bytes()
    result=build(root=root,snapshot=SNAP,universe_name=NAME,
                 start='2004-01-01',end='2004-12-31')
    assert result['selected']==1
    assert result['excluded']==2
    assert result['automatically_excluded_prelisting_stock_years']==1
    assert result['automatically_excluded_prelisting_examples']==['BAJAJ-AUTO/2004']
    assert result['raw_kaggle_stock_years']==0
    assert p.read_bytes()==original  # do not silently rewrite operator decisions
    assert not bar_path(root,'INE917I01010',2004,'research_v2').exists()


def test_undecided_entirely_prelisting_year_is_safely_excluded(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='BAJAJ-AUTO','decision']='requires_review'
    d.loc[d.symbol=='TCS','decision']='must_exclude'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.to_csv(p,index=False)
    result=build(root=root,snapshot=SNAP,universe_name=NAME,
                 start='2004-01-01',end='2004-12-31')
    assert result['automatically_excluded_prelisting_stock_years']==1
    assert result['excluded']==2 and result['selected']==1


def test_listing_year_with_only_prelisting_prices_is_auto_excluded(tmp_path):
    root=setup(tmp_path)
    # All TCS 2004 rows precede TCS's verified 2004-08-25 listing.
    path=bar_path(root,'INE467B01029',2004,'yahoo_finance')
    path.unlink()
    save_year(root,prices('TCS','INE467B01029',2004,days=['2004-06-01']),
              isin='INE467B01029',year=2004,provider='yahoo_finance')
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='approved_adjusted'
    d.loc[d.symbol=='TCS','trim_prelisting']='true'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.to_csv(p,index=False)
    result=build(root=root,snapshot=SNAP,universe_name=NAME,
                 start='2004-01-01',end='2004-12-31')
    assert result['selected']==1
    assert result['automatically_excluded_prelisting_stock_years']==1
    assert result['automatically_excluded_prelisting_examples']==['TCS/2004']
    assert not bar_path(root,'INE467B01029',2004,'research_v2').exists()


def test_partial_listing_year_requires_explicit_trim_approval(tmp_path):
    root=setup(tmp_path)
    audit(root=root,snapshot=SNAP,universe_name=NAME,start='2004-01-01',end='2004-12-31')
    p=manifest(root);d=pd.read_csv(p,dtype=str,keep_default_na=False)
    d.loc[d.symbol=='TCS','decision']='approved_adjusted'
    d.loc[d.symbol=='ITC','decision']='approved_adjusted'
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='trim_prelisting=true'):
        build(root=root,snapshot=SNAP,universe_name=NAME,
              start='2004-01-01',end='2004-12-31')
