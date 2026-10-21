"""Offline regression tests for user-extracted Kaggle Nifty-50 history."""
import json
from pathlib import Path

import pandas as pd
import pytest

from marketdata.kaggle_nifty50 import audit, read_kaggle
from marketdata.shared_store import bar_path, save_year


SNAPSHOT = '2026-10-21'
SYMBOLS = [('M&M', 'INE101A01026'), ('TATAMOTORS', 'INE155A01022')]


def setup_files(tmp_path):
    root = tmp_path/'market'; basket='testbasket'
    u = root/'universes'/basket
    u.mkdir(parents=True)
    pd.DataFrame([{'symbol':s, 'isin':i,'yahoo_ticker':s+'.NS'} for s,i in SYMBOLS]).to_csv(u/f'{SNAPSHOT}.csv',index=False)
    k = root/'kaggle';k.mkdir()
    pd.DataFrame([{'Symbol':s,'ISIN Code':i} for s,i in SYMBOLS]).to_csv(k/'stock_metadata.csv',index=False)
    raw = pd.DataFrame([
        {'Date':'2016-01-04','Symbol':'TELCO','Series':'EQ','Open':100,'High':105,'Low':95,'Close':102,'Volume':1200},
        {'Date':'2016-01-05','Symbol':'TATAMOTORS','Series':'EQ','Open':103,'High':106,'Low':98,'Close':104,'Volume':1800},
    ])
    raw.to_csv(k/'TATAMOTORS.csv',index=False)
    pd.DataFrame([{'Date':'2016-01-04','Symbol':'M&M','Series':'EQ',
                    'Open':201,'High':220,'Low':200,'Close':210,'Volume':500}]).to_csv(k/'MM.csv',index=False)
    existing = pd.DataFrame([{'date':'2016-01-04','symbol':'TATAMOTORS','isin':SYMBOLS[1][1],
        'open':10.,'high':10.5,'low':9.5,'close':10.2,'volume':1200}])
    yahoo_file=save_year(root,existing,isin=SYMBOLS[1][1],year=2016,provider='yahoo_finance')
    return root,k,yahoo_file


def test_audit_preserves_yahoo_and_canonical(tmp_path):
    root,k,yahoo_path=setup_files(tmp_path)
    original=yahoo_path.read_bytes()
    first=audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket')
    assert first['checked']==2 and first['errors']==0
    assert first['kaggle_only_dates']==2  # M&M entirely absent; TATAMOTORS one day absent
    assert first['yahoo_adjusted_partitions_modified'] is False
    assert yahoo_path.read_bytes()==original
    assert bar_path(root,SYMBOLS[0][1],2016,'kaggle_raw').is_file()
    compare=pd.read_csv(root/'reports'/'kaggle_raw'/f'testbasket_{SNAPSHOT}_comparisons.csv')
    assert compare.loc[compare.symbol.eq('TATAMOTORS'),'close_differences_over_tolerance'].iloc[0]==1
    missing=pd.read_csv(root/'reports'/'kaggle_raw'/f'testbasket_{SNAPSHOT}_missing_yahoo_dates.csv')
    assert set(missing.reason)=={'missing_yahoo_partition','missing_yahoo_session'}
    assert not (root/'candles'/'research_ohlcv').exists()
    # Existing cached Kaggle stock/year passes identity and content check, not silently overwritten.
    second=audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket')
    assert second['errors']==0 and second['kaggle_only_dates']==2
    assert yahoo_path.read_bytes()==original


def test_historic_tickers_and_file_alias(tmp_path):
    root,k,_=setup_files(tmp_path)
    frame,aliases=read_kaggle(k/'TATAMOTORS.csv',symbol='TATAMOTORS',isin=SYMBOLS[1][1])
    assert aliases == ['TATAMOTORS','TELCO']
    assert frame.symbol.nunique()==1 and frame.symbol.iloc[0]=='TATAMOTORS'
    assert frame.date.iloc[0] == pd.Timestamp('2016-01-04')
    coverage=audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket')['report_directory']
    df=pd.read_csv(Path(coverage)/f'testbasket_{SNAPSHOT}_coverage.csv')
    assert df.loc[df.symbol.eq('M&M'),'kaggle_file'].iloc[0]=='MM.csv'


def test_isin_mismatch_blocks_wrong_identity(tmp_path):
    root,k,yahoo_path=setup_files(tmp_path)
    df=pd.read_csv(k/'stock_metadata.csv',dtype=str)
    df.loc[df.Symbol.eq('TATAMOTORS'),'ISIN Code']='INE999A01011'
    df.to_csv(k/'stock_metadata.csv',index=False)
    result=audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket')
    assert result['isin_mismatches']==1
    assert not bar_path(root,SYMBOLS[1][1],2016,'kaggle_raw').exists()
    assert yahoo_path.is_file()


def test_invalid_ohlc_fails_without_damaging_existing_yahoo(tmp_path):
    root,k,yahoo_path=setup_files(tmp_path)
    df=pd.read_csv(k/'TATAMOTORS.csv')
    df.loc[0,'High']=1
    df.to_csv(k/'TATAMOTORS.csv',index=False)
    result=audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket')
    assert result['errors']==1
    err=pd.read_csv(root/'reports'/'kaggle_raw'/f'testbasket_{SNAPSHOT}_errors.csv')
    assert err.symbol.tolist()==['TATAMOTORS']
    assert yahoo_path.is_file()
    with pytest.raises(ValueError):
        audit(root=root,snapshot=SNAPSHOT,universe_name='testbasket',price_tolerance_pct=-1)
