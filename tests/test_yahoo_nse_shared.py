"""Offline regression tests for the no-login Yahoo + NSE shared daily data route."""
import json
import numpy as np
import pandas as pd
import pytest

from marketdata.shared_store import load_universe, load_daily, data_root, bar_path
from marketdata.yahoo_nifty50 import collect, parse_constituents, normalize_yahoo
from marketdata.nse_daily import import_day, parse_nse_bhavcopy
from models.nifty50_stage1 import verify_download_audits


def csv_universe(n=50):
    return 'Symbol,ISIN Code\n' + ''.join(f'S{i:02d},INE{i:09d}\n' for i in range(n))


def fake_downloader(ticker, *, start, end, **kwargs):
    if ticker == 'S02.NS':
        return pd.DataFrame()
    idx = pd.DatetimeIndex([f'{start[:4]}-01-02',f'{start[:4]}-01-03'])
    return pd.DataFrame({'Open':[100,110], 'High':[115,114], 'Low':[99,109],
                         'Close':[109,113], 'Volume':[1000,1200]}, index=idx)


def test_yahoo_stages_save_and_cache_without_login(tmp_path):
    csv = tmp_path / 'nifty.csv'; csv.write_text(csv_universe())
    root = tmp_path / 'shared'
    run = dict(root=root,start='2015-01-01',end='2015-12-31',asof='2026-10-18',
               universe_csv=csv,downloader=fake_downloader,pause=0)
    first = collect(**run)
    assert first['downloaded']==49 and first['no_data_unverified']==1 and first['errors']==0
    assert first['universe_name']=='nifty50'
    assert bar_path(root,'INE000000000',2015).is_file()
    snapshot = load_universe(root, '2026-10-18')
    assert len(snapshot)==50 and snapshot.yahoo_ticker.iloc[0]=='S00.NS'
    assert len(load_daily(root,snapshot,'2015-01-01','2015-12-31'))==98
    # Simulate Windows CRLF from a previously saved CSV snapshot.
    snapshot_file=root/'universes'/'nifty50'/'2026-10-18.csv'
    snapshot_file.write_bytes(snapshot_file.read_bytes().replace(b'\n',b'\r\n'))
    calls=[]
    def no_download(*a,**k):
        calls.append(a)
        raise AssertionError('Cache must be reused')
    second=collect(**{**run,'downloader':no_download})
    assert len(calls)==3 and second['cached']==49 and second['errors']==1
    # An empty (possibly pre-listing) year is deliberately retried, not cached as fact.


def test_failure_is_not_silently_treated_as_prelisting(tmp_path):
    csv=tmp_path/'nifty.csv';csv.write_text(csv_universe())
    def fail(symbol,**k):
        if symbol=='S03.NS': raise TimeoutError('rate limited')
        return fake_downloader(symbol,**k)
    summary=collect(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',
                    asof='2026-10-18',universe_csv=csv,downloader=fail,pause=0)
    assert summary['errors']==1 and summary['no_data_unverified']==1
    audit=pd.read_csv(summary['status_csv'])
    assert audit.loc[audit.symbol.eq('S03'),'status'].iloc[0]=='error'


def test_future_custom_universe_and_no_cross_universe_cache_changes(tmp_path):
    source=tmp_path/'custom.csv';source.write_text(csv_universe(3))
    report=collect(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',
                   asof='2026-10-18',universe_csv=source,universe_name='smallcap_test',
                   downloader=fake_downloader,pause=0)
    assert report['universe_size']==3
    assert (tmp_path/'shared'/'universes'/'smallcap_test'/'2026-10-18.csv').is_file()
    assert len(load_universe(tmp_path/'shared','2026-10-18','smallcap_test'))==3
    with pytest.raises(ValueError,match='outside'):
        data_root('marketdata/local')
    with pytest.raises(ValueError,match='Universe name'):
        collect(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',
                universe_name='../escapes',universe_csv=source,downloader=fake_downloader,pause=0)


def test_parse_yahoo_multiindex_stock_prices():
    idx=pd.DatetimeIndex(['2016-03-01 00:00:00+05:30'])
    cols=pd.MultiIndex.from_product([['Open','High','Low','Close','Volume'],['S00.NS']])
    bars=normalize_yahoo(pd.DataFrame([[11.,12.,10.,11.5,1000]],index=idx,columns=cols),
                         symbol='S00',isin='INE000000000',year=2016)
    assert len(bars)==1 and bars['date'].iloc[0]==pd.Timestamp('2016-03-01')
    assert bars['volume'].iloc[0]==1000


def test_nse_bhavcopy_import_is_non_destructive(tmp_path):
    source=tmp_path/'small.csv';source.write_text(csv_universe(2))
    root=tmp_path/'shared';asof='2026-10-18'
    collect(root=root,start='2015-01-01',end='2015-12-31',asof=asof,
            universe_csv=source,universe_name='testbasket',downloader=fake_downloader,pause=0)
    bhav=(
        'TckrSymb,ISIN,SctySrs,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n'
        'S00,INE000000000,EQ,02-Jan-2015,101,120,90,118,2200\n'
        'S01,INE000000001,EQ,02-Jan-2015,102,121,91,119,2300\n'
    ).encode()
    path=tmp_path/'bhav.csv';path.write_bytes(bhav)
    imported=import_day(root=root,snapshot=asof,session='2015-01-02',archive_file=path,
                        universe_name='testbasket')
    assert imported['stocks_in_daily_archive']==2
    assert imported['yahoo_overlap_rows']==2
    assert bar_path(root,'INE000000000',2015,'nse_raw').is_file()
    assert pd.read_csv(bar_path(root,'INE000000000',2015,'yahoo_finance')).close.iloc[0]==109.
    again=import_day(root=root,snapshot=asof,session='2015-01-02',archive_file=path,
                     universe_name='testbasket')
    assert again['new_or_updated_partitions']==0
    with pytest.raises(ValueError,match='ISIN mismatch'):
        parse_nse_bhavcopy(bhav.replace(b'INE000000001',b'INE000000099'),
                           session='2015-01-02',universe=load_universe(root,asof,'testbasket'))


def test_constituents_validate_duplicates_and_expected_50():
    with pytest.raises(ValueError,match='count mismatch'):
        parse_constituents(csv_universe(3))
    assert len(parse_constituents(csv_universe(3),expected_count=None))==3
    with pytest.raises(ValueError,match='duplicate'):
        parse_constituents('Symbol,ISIN Code\nS00,INE000000000\nS00,INE000000001\n',expected_count=None)


def test_model_uses_custom_universe_and_never_tunes_on_2017_2019(tmp_path):
    from marketdata.shared_store import save_year
    from models.nifty50_stage1 import train_and_validate
    root=tmp_path/'market-data'; snapshot='2026-10-18'; universe_name='smallcap_test'
    folder=root/'universes'/universe_name; folder.mkdir(parents=True)
    symbols=[f'S{i:02d}' for i in range(3)]
    isins=[f'INE{i:09d}' for i in range(3)]
    pd.DataFrame({'symbol':symbols,'isin':isins,'yahoo_ticker':[s+'.NS' for s in symbols]}).to_csv(folder/f'{snapshot}.csv',index=False)
    report_folder=root/'reports'/'yahoo_finance';report_folder.mkdir(parents=True)
    for first,last,n in [('2000-01-01','2016-12-31',17),('2017-01-01','2019-12-31',3)]:
        (report_folder/f'{universe_name}_{snapshot}_{first}_{last}_summary.json').write_text(json.dumps({
            'provider':'yahoo_finance','universe_name':universe_name,'snapshot':snapshot,
            'start':first,'end':last,'universe_size':3,
            'requests_or_cache_checks':3*n,'errors':0,'no_data_unverified':3*n-6 if n==17 else 0}))
    dates=pd.bdate_range('2014-01-01','2019-12-31')
    for i in range(3):
        c=100+np.arange(len(dates))*(i+1)*.08+np.sin(np.arange(len(dates))/17+i)*.7
        frame=pd.DataFrame({'date':dates,'symbol':symbols[i],'isin':isins[i],
             'open':c,'high':c+2,'low':c-2,'close':c,'volume':500000+i*100})
        for year,part in frame.groupby(frame.date.dt.year):
            save_year(root,part,isin=isins[i],year=int(year))
    summary=train_and_validate(root=root,snapshot=snapshot,universe_name=universe_name,
                               output=tmp_path/'results',horizon=5,top_k=2,estimators=10,
                               allow_unverified_adjustments=True)
    assert summary['train_sessions']>252 and summary['holdout_sessions']>250
    assert summary['training_label_end_latest']<'2017-01-01'
    assert summary['results']['lightgbm']['completed_trades']>0
    assert summary['live_execution_enabled'] is False
    assert (tmp_path/'results'/'ranker.txt').is_file()
    (report_folder/f'{universe_name}_{snapshot}_2017-01-01_2019-12-31_summary.json').unlink()
    with pytest.raises(ValueError,match='Missing Yahoo download audit'):
        verify_download_audits(root,snapshot,universe_name=universe_name)
