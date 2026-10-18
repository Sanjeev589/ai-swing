"""No network, secrets or real order calls. Synthetic checks for staged Upstox research."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from marketdata.shared_store import data_root, load_daily, save_year, validate_bars
from marketdata.upstox_v3 import candle_url, collect, parse_candles, parse_universe
from models.upstox_nifty50_stage1 import _paper_cohorts, train_and_validate


def fake_universe():
    symbols = [f'SYM{i:02d}' for i in range(50)]
    csv_text = 'Symbol\n'+'\n'.join(symbols)+'\n'
    contracts = [dict(segment='NSE_EQ', instrument_type='EQ', trading_symbol=s,
                      isin=f'INE{i:09d}', instrument_key=f'NSE_EQ|INE{i:09d}') for i, s in enumerate(symbols)]
    return csv_text, contracts


def test_match_only_current_nse_eq_and_check_50():
    text, contracts = fake_universe()
    contracts.append({**contracts[0], 'instrument_type':'BE'})
    mapped = parse_universe(text,contracts)
    assert len(mapped)==50 and mapped.symbol.is_unique
    with pytest.raises(ValueError, match='uniquely map'):
        parse_universe(text,contracts[:-2])


def test_v3_daily_dates_and_strict_candles():
    item=SimpleNamespace(symbol='AAA',isin='INE000000001',instrument_key='NSE_EQ|INE000000001')
    data = parse_candles([['2016-06-01T00:00:00+05:30',10,12,9,11,500,0]],item,2016,date(2016,1,1),date(2016,12,31))
    assert data.loc[0,'date']==pd.Timestamp('2016-06-01')
    assert data.loc[0,'volume']==500
    assert 'NSE_EQ%7CINE' in candle_url(item.instrument_key,date(2016,1,1),date(2016,12,31))
    with pytest.raises(ValueError,match='Out-of-range'):
        parse_candles([['2017-01-01T00:00:00+05:30',10,12,9,11,500]],item,2016,date(2016,1,1),date(2016,12,31))
    with pytest.raises(ValueError,match='Inconsistent'):
        parse_candles([['2016-06-01T00:00:00+05:30',10,8,9,11,500]],item,2016,date(2016,1,1),date(2016,12,31))


def test_shared_store_excludes_repository_and_preserves_data(tmp_path):
    root=tmp_path/'market-data'
    assert data_root(root)==root
    year=2015
    df=pd.DataFrame({'date':['2015-01-01','2015-01-02'], 'symbol':['AAA','AAA'],
        'isin':['INE000000001']*2, 'instrument_key':['NSE_EQ|INE000000001']*2,
        'open':[10.,11.], 'high':[12.,13.], 'low':[9.,10.], 'close':[11.,12.],
        'volume':[100.,200.], 'open_interest':[0,0]})
    dest=save_year(root,df,isin='INE000000001',year=year)
    assert dest.exists()
    universe=pd.DataFrame({'symbol':['AAA'], 'isin':['INE000000001'],'instrument_key':['NSE_EQ|INE000000001']})
    loaded=load_daily(root,universe,'2015-01-01','2015-12-31')
    assert len(loaded)==2
    with pytest.raises(ValueError,match='Duplicate'):
        validate_bars(pd.concat([df,df],ignore_index=True))
    assert save_year(root,df,isin='INE000000001',year=year).read_bytes()==dest.read_bytes()


def test_download_never_starts_without_access_token(tmp_path, monkeypatch):
    monkeypatch.delenv('UPSTOX_ACCESS_TOKEN',raising=False)
    with pytest.raises(ValueError,match='UPSTOX_ACCESS_TOKEN'):
        collect(root=tmp_path,start='2000-01-01',end='2016-12-31')
    with pytest.raises(ValueError,match='full-year'):
        collect(root=tmp_path,start='2015-02-01',end='2016-12-31',token='test')


def test_report_status_cached_empty_and_errors(tmp_path, monkeypatch):
    text, contracts = fake_universe()
    csv=tmp_path/'current.csv';csv.write_text(text)
    instr=tmp_path/'nse.json.gz';instr.write_bytes(gzip.compress(json.dumps(contracts).encode()))
    import marketdata.upstox_v3 as module
    class Session:
        def get(self,url,headers=None,timeout=None):
            # Unit-test fake API, never transmit token
            return SimpleNamespace(status_code=200,headers={},raise_for_status=lambda:None,
                json=lambda:{'status':'success','data':{'candles':[['2015-05-04T00:00:00+05:30',10,12,9,11,500,0]]}})
    result=collect(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',token='test',
                   asof='2026-10-18',universe_csv=str(csv),instrument_json=str(instr),session=Session(),pause=0)
    assert result['downloaded']==50 and result['errors']==0
    again=collect(root=tmp_path/'shared',start='2015-01-01',end='2015-12-31',token='test',
                   asof='2026-10-18',universe_csv=str(csv),instrument_json=str(instr),session=Session(),pause=0)
    assert again['cached']==50


def test_no_holdout_tuning_or_open_price_leakage():
    dates=pd.bdate_range('2017-01-02', periods=25)
    rows=[]
    for day in dates:
        for symbol,score in [('AAA',3.),('BBB',1.)]:
            loc=dates.get_loc(day)
            if loc+6>=len(dates): continue
            rows.append({'date':day,'symbol':symbol,'entry_date':dates[loc+1],
                         'label_end_date':dates[loc+6],'target_return':.04 if symbol=='AAA' else -.04,
                         'score':score,'rule_score':score})
    frame=pd.DataFrame(rows)
    trades, metrics=_paper_cohorts(frame,rank_by='score',sessions=dates,top_k=1,horizon=5,side_cost_bps=25)
    assert metrics['winning_trades']==metrics['completed_trades']
    assert metrics['completed_trades']==4
    assert all(trades.entry_date > trades.signal_date)


def test_exploratory_training_requires_explicit_action(tmp_path):
    with pytest.raises(ValueError,match='unverified'):
        train_and_validate(root=tmp_path,snapshot='2026-10-18')


def test_full_model_fixed_holdout_on_synthetic_series(tmp_path):
    from marketdata.shared_store import save_year
    from models.upstox_nifty50_stage1 import train_and_validate
    root=tmp_path/'market-data'; snapshot='2026-10-18'
    directory=root/'universes'/'nifty50'; directory.mkdir(parents=True)
    symbols=[f'S{i:02d}' for i in range(50)]
    instruments=[f'INE{i:09d}' for i in range(50)]
    pd.DataFrame({'symbol':symbols,'isin':instruments,'instrument_key':[f'NSE_EQ|{x}' for x in instruments]}).to_csv(directory/f'{snapshot}.csv',index=False)
    audits=root/'reports'/'upstox_v3'; audits.mkdir(parents=True)
    for begin,end,count in [('2000-01-01','2016-12-31',850),('2017-01-01','2019-12-31',150)]:
        (audits/f'nifty50_{snapshot}_{begin}_{end}_summary.json').write_text(json.dumps({
            'start':begin,'end':end,'snapshot':snapshot,'universe_size':50,
            'requests_or_cache_checks':count,'errors':0,'no_data':count//2}))
    # Only 3 of today's 50 trade in this synthetic history. Real data may have IPO gaps.
    dates=pd.bdate_range('2014-01-01','2019-12-31')
    for i in range(3):
        close=(100+np.arange(len(dates))*(i+1)*.08+np.sin(np.arange(len(dates))/17+i)*.7)
        df=pd.DataFrame({'date':dates,'symbol':symbols[i],'isin':instruments[i],
           'instrument_key':f'NSE_EQ|{instruments[i]}','open':close,'high':close+2,
           'low':close-2,'close':close,'volume':500000+i*100,'open_interest':0})
        for year,part in df.groupby(df.date.dt.year):
            save_year(root,part,isin=instruments[i],year=int(year))
    report=train_and_validate(root=root,snapshot=snapshot,output=tmp_path/'models',
                              horizon=5,top_k=2,estimators=10,allow_unverified_corporate_actions=True)
    assert report['training_label_end_latest']<'2017-01-01'
    assert report['holdout_sessions']>100
    assert report['live_execution_enabled'] is False
    assert report['results']['lightgbm']['completed_trades']>0
    assert (tmp_path/'models'/'ranker.txt').exists()
    assert (tmp_path/'models'/'holdout_trades.csv').exists()