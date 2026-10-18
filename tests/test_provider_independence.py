"""Offline tests for user-controlled retry suppression and multi-vendor OHLCV."""
import csv
import json

import pandas as pd
import pytest

from marketdata.yahoo_nifty50 import collect
from marketdata.research_store import build_view, import_external
from marketdata.shared_store import bar_path, load_daily, load_universe


def universe_csv(count):
    return 'Symbol,Company Name,ISIN Code\n' + ''.join(
        f'S{i:03d},Stock {i:03d},INE{i:09d}\n' for i in range(count))


def prices(ticker, *, start, end, **_):
    d = pd.DatetimeIndex([f'{start[:4]}-02-02', f'{start[:4]}-02-03'])
    return pd.DataFrame({'Open':[100.,101.], 'High':[105.,105.],
                         'Low':[99.,100.], 'Close':[104.,103.], 'Volume':[1000,1200]},index=d)


def setup_collection(tmp_path, count=3, universe_name='new_universe'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    csv_file = tmp_path/'symbols.csv'
    csv_file.write_text(universe_csv(count))
    options=dict(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',
                 asof='2026-10-18',universe_name=universe_name,universe_csv=csv_file,pause=0)
    return options


def test_legacy_issue_log_migrates_preserves_false_and_manual_true(tmp_path):
    options = setup_collection(tmp_path)
    seen=[]
    def failure(ticker,**kw):
        seen.append(ticker)
        if ticker=='S001.NS':raise TimeoutError('Yahoo 429')
        if ticker=='S002.NS':return pd.DataFrame()
        return prices(ticker,**kw)
    first=collect(**options,downloader=failure)
    assert (first['downloaded'],first['errors'],first['no_data_unverified'])==(1,1,1)
    error_path=tmp_path/'shared'/'reports'/'yahoo_finance'/'new_universe_download_errors.csv'
    with error_path.open(newline='') as f:
        rows=list(csv.DictReader(f)); assert len(rows)==2
    assert all(r['ignore']=='false' for r in rows)
    # Simulate old log from the previous application version lacking ignore.
    with error_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='ignore'])
        writer.writeheader(); writer.writerows([{k:v for k,v in row.items() if k!='ignore'} for row in rows])
    # Migration to editable log, without changing user data or downloading anything.
    def repaired(ticker,**kw):
        return prices(ticker,**kw)
    migrated=collect(**options,downloader=repaired)
    assert migrated['cached']==1 and migrated['downloaded']==2
    with error_path.open(newline='') as f: rows=list(csv.DictReader(f))
    assert 'ignore' in rows[0] and len(rows)==2
    # New failures appended to old rows, then mark one as permanently ignored.
    alt=setup_collection(tmp_path/'other')
    collect(**alt,downloader=failure)
    alt_error=tmp_path/'other'/'shared'/'reports'/'yahoo_finance'/'new_universe_download_errors.csv'
    with alt_error.open(newline='') as f:edited=list(csv.DictReader(f))
    edited[0]['ignore']='TRUE'
    with alt_error.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=edited[0].keys());writer.writeheader();writer.writerows(edited)
    calls=[]
    def retries(ticker,**kw):
        calls.append(ticker);return prices(ticker,**kw)
    second=collect(**alt,downloader=retries)
    assert calls==['S002.NS']
    assert second['cached']==1 and second['ignored']==1 and second['downloaded']==1
    assert second['errors']==0
    with alt_error.open(newline='') as f:observed=list(csv.DictReader(f))
    assert observed[0]['ignore']=='TRUE' and len(observed)==2
    # Ignore state must remain independent per universe and vendor.
    assert len(list(error_path.open().readlines()))==3


def test_manual_ignore_false_remains_retryable(tmp_path):
    options=setup_collection(tmp_path)
    def failing(ticker,**kw):
        if ticker=='S001.NS':raise RuntimeError('temporary')
        return prices(ticker,**kw)
    collect(**options,downloader=failing)
    p=tmp_path/'shared'/'reports'/'yahoo_finance'/'new_universe_download_errors.csv'
    with p.open(newline='') as f: rows=list(csv.DictReader(f))
    rows[0]['ignore']='false'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    attempts=[]
    def recovered(ticker,**kw):
        attempts.append(ticker);return prices(ticker,**kw)
    result=collect(**options,downloader=recovered)
    assert result['cached']==2 and result['downloaded']==1 and attempts==['S001.NS']


def test_nifty500_is_supported_for_future_universes(tmp_path):
    options=setup_collection(tmp_path,count=5,universe_name='nifty500')
    out=collect(**options,downloader=prices)
    assert out['universe_size']==5 and out['downloaded']==5 and out['errors']==0
    loaded=load_universe(tmp_path/'shared','2026-10-18',universe_name='nifty500')
    assert len(loaded)==5


def test_external_provider_canonical_source_and_idempotence(tmp_path):
    options=setup_collection(tmp_path,count=3)
    def fails_some(ticker,**kw):
        if ticker=='S001.NS':raise RuntimeError('no Yahoo data')
        return prices(ticker,**kw)
    result=collect(**options,downloader=fails_some)
    assert result['errors']==1
    external=tmp_path/'external.csv'
    external.write_text('date,symbol,open,high,low,close,volume\n'
                        '2015-03-02,S001,140,145,138,143,2000\n'
                        '2015-03-03,S001,145,148,142,146,2500\n')
    entry=import_external(root=options['root'],csv_path=external,
                          source='ext_research_vendor',snapshot='2026-10-18',universe_name='new_universe',
                          acknowledge_adjusted=True)
    assert entry['source']=='ext_research_vendor'
    assert bar_path(options['root'],'INE000000001',2015,'ext_research_vendor').is_file()
    combined=build_view(root=options['root'],snapshot='2026-10-18',universe_name='new_universe',
                        start='2015-01-01',end='2015-12-31',
                        sources=['yahoo_finance','ext_research_vendor'])
    assert combined['selected']==3 and combined['errors']==0 and combined['missing']==0
    original=bar_path(options['root'],'INE000000001',2015,'research_ohlcv')
    manifest=json.loads(original.with_name('2015.source.json').read_text())
    assert manifest['source']=='ext_research_vendor'
    snapshot=load_universe(options['root'],'2026-10-18',universe_name='new_universe')
    training=load_daily(options['root'],snapshot,'2015-01-01','2015-12-31',provider='research_ohlcv')
    assert set(training.symbol)=={'S000','S001','S002'}
    # Canonical view is stable even if vendor priority is changed later.
    again=build_view(root=options['root'],snapshot='2026-10-18',universe_name='new_universe',
                     start='2015-01-01',end='2015-12-31',
                     sources=['ext_research_vendor','yahoo_finance'])
    assert again['cached']==3
    assert json.loads(original.with_name('2015.source.json').read_text())['source']=='ext_research_vendor'
    assert bar_path(options['root'],'INE000000000',2015,'yahoo_finance').is_file()
    assert not bar_path(options['root'],'INE000000001',2015,'yahoo_finance').is_file()


def test_source_validation_and_unaudited_raw_source_rejected(tmp_path):
    options=setup_collection(tmp_path)
    collect(**options,downloader=prices)
    csv_file=tmp_path/'bad.csv'
    csv_file.write_text('date,symbol,open,high,low,close,volume\n2015-02-02,S001,100,105,99,103,1000\n')
    with pytest.raises(ValueError,match='yahoo_finance or ext_'):
        import_external(root=options['root'],csv_path=csv_file,source='nse_raw',
                        snapshot='2026-10-18',universe_name='new_universe',acknowledge_adjusted=True)
    with pytest.raises(ValueError,match='yahoo_finance or ext_'):
        build_view(root=options['root'],snapshot='2026-10-18',universe_name='new_universe',
                   start='2015-01-01',end='2015-12-31',sources=['nse_raw'])
    bad=tmp_path/'symbol_wrong.csv'
    bad.write_text('date,symbol,open,high,low,close,volume\n2015-02-02,S999,100,105,99,103,1000\n')
    with pytest.raises(ValueError,match='not part'):
        import_external(root=options['root'],csv_path=bad,source='ext_other',
                        snapshot='2026-10-18',universe_name='new_universe',acknowledge_adjusted=True)