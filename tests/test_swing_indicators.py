from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from features.swing_indicators import calculate_features, wilder, supertrend
from features.generate_features import generate
from data.feature_inputs import read_benchmark, read_stock_zip


def inputs(count=320, flat=False):
    dates=pd.bdate_range('2018-01-01',periods=count)
    close=np.full(count,100.) if flat else np.arange(count,dtype=float)+100
    stock=pd.DataFrame({'date':dates,'symbol':'TEST','open':close,'high':close+1,'low':close-1,'close':close,'volume':100})
    index=pd.DataFrame({'date':dates,'nifty500_close':np.full(count,1000.)})
    return stock,index


def test_wilder_seed_and_reset():
    result=wilder(pd.Series([1.,2.,3.,6.,np.nan,4.,5.,6.]),3)
    assert result.iloc[2]==2
    assert result.iloc[3]==pytest.approx(10/3)
    assert result.iloc[4:7].isna().all()
    assert result.iloc[7]==5


def test_trending_indicators_and_market_relative_strength():
    stock,index=inputs()
    result=calculate_features(stock,index)
    assert result.rsi_14.iloc[14]==100
    assert result.atr_14.iloc[13]==2
    assert result.adx_14.iloc[27]==100
    assert result.adx_14.iloc[:27].isna().all()
    assert result.high_252.iloc[250:].iloc[0]!=result.high_252.iloc[250:].iloc[0]
    assert result.high_252.iloc[251]==352
    assert result.relative_strength_20.iloc[20]==pytest.approx(.2)
    assert result.relative_strength_60.iloc[60]==pytest.approx(.6)
    assert result.volume_ratio_20.iloc[19]==1
    assert result.bb_upper_20_2.iloc[19]==pytest.approx(109.5+2*np.std(np.arange(100,120)))
    assert result.ema_9.iloc[-1]==pytest.approx(stock.close.ewm(span=9,adjust=False).mean().iloc[-1])
    assert result.supertrend_direction.iloc[-1]==1
    assert result.features_ready.iloc[251:].all()


def test_flat_and_zero_volume():
    stock,index=inputs(flat=True); stock['volume']=0
    stock['high']=stock['low']=100
    result=calculate_features(stock,index)
    assert result.rsi_14.iloc[14]==50
    assert result.adx_14.iloc[27]==0
    assert result.atr_14.iloc[13]==0
    assert result.volume_ratio_20.isna().all()
    assert result.bb_width.iloc[19]==0


def test_causality_symbol_isolation_and_missing_session():
    stock,index=inputs()
    original=calculate_features(stock,index)
    prefix=calculate_features(stock.iloc[:280],index.iloc[:280])
    pd.testing.assert_frame_equal(original.iloc[:280].reset_index(drop=True),prefix)
    second=stock.assign(symbol='OTHER',open=stock.open*2,high=stock.high*2,low=stock.low*2,close=stock.close*2)
    both=calculate_features(pd.concat([stock,second]),index)
    pd.testing.assert_frame_equal(original,both[both.symbol.eq('TEST')].reset_index(drop=True))
    gap=calculate_features(stock.drop(index=270),index)
    assert gap.loc[gap.date.eq(stock.date.iloc[271]),'ema_9'].isna().all()
    assert gap.loc[gap.date.eq(stock.date.iloc[290]),'relative_strength_20'].isna().all()
    assert gap.loc[gap.date.eq(stock.date.iloc[280]),'relative_strength_20'].notna().all()


def test_supertrend_reversal():
    close=pd.Series([10.,11.,15.,16.,5.,4.])
    line,direction=supertrend(close+1,close-1,close,pd.Series([1.]*6),3)
    assert direction.tolist()==[-1,-1,1,1,-1,-1]
    assert line.iloc[2]==12
    assert line.iloc[4]==8


def test_official_parsers():
    payload=b'Index Name,Index Date,Closing Index Value\nNifty 500,02-01-2019,9000\nNifty 500 Shariah,02-01-2019,100\n'
    assert read_benchmark(payload,'2019-01-02')['nifty500_close']==9000
    with pytest.raises(ValueError,match='date mismatch'): read_benchmark(payload,'2019-01-03')
    buffer=BytesIO()
    with ZipFile(buffer,'w') as archive:
        archive.writestr('test.csv','SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,TOTTRDQTY,TIMESTAMP\nTEST,RR,10,12,9,11,100,02-JAN-2019\n')
    assert read_stock_zip(buffer.getvalue(),'2019-01-02',{'TEST'}).high.iloc[0]==12
    buffer=BytesIO()
    with ZipFile(buffer,'w') as archive:
        archive.writestr('test.csv','SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,TOTTRDQTY,TIMESTAMP\nTEST,EQ,10,12,9,11,100,13-Jul-20\n')
    assert read_stock_zip(buffer.getvalue(),'2020-07-13',{'TEST'}).date.iloc[0]==pd.Timestamp('2020-07-13')


def test_generation_preserves_year_boundary(tmp_path):
    stock,index=inputs(600)
    source=tmp_path/'inputs'; source.mkdir()
    stock.to_csv(source/'ohlcv_2018.csv.gz',index=False)
    index.to_csv(source/'nifty500_index.csv',index=False)
    manifest=generate(source,tmp_path/'output',2019,2020)
    annual=pd.read_csv(tmp_path/'output/features_2019.csv.xz',parse_dates=['date'])
    expected=calculate_features(stock,index)
    expected=expected[expected.date.dt.year.eq(2019)].reset_index(drop=True)
    np.testing.assert_allclose(annual.ema_50,expected.ema_50,rtol=1e-8)
    assert manifest['files'][0]['ready_rows']>0


def test_market_relative_strength_uses_benchmark_and_macd_signal():
    stock,index=inputs()
    index['nifty500_close']=np.arange(len(index))*10+1000
    result=calculate_features(stock,index)
    assert result.relative_strength_20.iloc[20]==pytest.approx(0)
    macd=stock.close.ewm(span=12,adjust=False,min_periods=12).mean()-stock.close.ewm(span=26,adjust=False,min_periods=26).mean()
    signal=macd.dropna().ewm(span=9,adjust=False,min_periods=9).mean()
    assert result.macd_signal.iloc[33]==pytest.approx(signal.loc[33])
    assert result.macd_histogram.iloc[33]==pytest.approx(macd.loc[33]-signal.loc[33])
    stock.loc[280:,'close']*=2
    stock.loc[280:,'open']=stock.loc[280:,'close']
    stock.loc[280:,'high']=stock.loc[280:,'close']+1
    stock.loc[280:,'low']=stock.loc[280:,'close']-1
    changed=calculate_features(stock,index)
    pd.testing.assert_frame_equal(result.iloc[:280],changed.iloc[:280])


def test_new_listing_and_inactive_invalid_range_remain_missing():
    stock,index=inputs()
    new=calculate_features(stock.iloc[200:],index)
    assert new.ema_50.iloc[:49].isna().all()
    assert new.high_252.isna().all()
    assert not new.features_ready.any()
    stock.loc[270,['open','high','low','volume']]=0
    result=calculate_features(stock,index)
    assert result.loc[270,['ema_9','atr_14','supertrend_10_3','high_252']].isna().all()
    assert result.loc[271,'ema_9']!=result.loc[271,'ema_9']


def test_benchmark_calendar_validation():
    stock,index=inputs()
    with pytest.raises(ValueError,match='benchmark calendar'):
        calculate_features(stock,index.iloc[1:])
    with pytest.raises(ValueError,match='benchmark dates'):
        calculate_features(stock,pd.concat([index,index.iloc[:1]]))
    index.loc[0,'date']+=pd.Timedelta(hours=12)
    with pytest.raises(ValueError,match='daily benchmark'):
        calculate_features(stock,index)
