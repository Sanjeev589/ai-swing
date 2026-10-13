import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from models.swing_dataset import MODEL_FEATURES,model_inputs,execution_labels,chronological_splits
from models.train_swing_ranker import ranking_query,new_ranker,score_frame,evaluate
from features.swing_indicators import calculate_features


def prices(count=320):
    dates=pd.bdate_range('2019-01-01',periods=count)
    frames=[]
    for number,symbol in enumerate(['AAA','BBB','CCC']):
        c=100+np.arange(count)*(number+1)*.1+np.sin(np.arange(count)/5+number)
        frames.append(pd.DataFrame({'date':dates,'symbol':symbol,'open':c,'high':c+1,'low':c-1,'close':c,'volume':1000}))
    return pd.concat(frames,ignore_index=True),dates


def test_calendar_aligned_labels_and_gaps():
    stock,dates=prices(30)
    labels=execution_labels(stock,dates,5)
    first=labels[labels.symbol.eq('AAA')].iloc[0]
    assert first.entry_date==dates[1] and first.label_end_date==dates[6]
    a=stock[stock.symbol.eq('AAA')].reset_index(drop=True)
    assert first.target_return==pytest.approx(a.open.iloc[6]/a.open.iloc[1]-1)
    missing=stock[~(stock.symbol.eq('AAA')&stock.date.eq(dates[3]))]
    gaps=execution_labels(missing,dates,5)
    assert gaps.loc[gaps.symbol.eq('AAA')&gaps.date.eq(dates[0]),'target_return'].isna().all()
    assert gaps.loc[gaps.symbol.eq('BBB')&gaps.date.eq(dates[0]),'target_return'].notna().all()
    assert labels[labels.date.isin(dates[-6:])].target_return.isna().all()
    before=execution_labels(stock[stock.date.le(dates[20])],dates[:21],5)
    pd.testing.assert_frame_equal(labels[labels.date.le(dates[14])].reset_index(drop=True),before[before.date.le(dates[14])].reset_index(drop=True))


def test_normalization_and_feature_allowlist():
    stock,dates=prices()
    saved=calculate_features(stock,pd.DataFrame({'date':dates,'nifty500_close':1000.}))
    original=model_inputs(saved)
    scaled=saved.copy()
    nominal=['open','high','low','close','ema_9','ema_21','ema_50','macd','macd_signal','macd_histogram','bb_middle_20','bb_upper_20_2','bb_lower_20_2','atr_14','supertrend_10_3','high_252']
    scaled[nominal]*=10
    scaled['target_return']=999
    other=model_inputs(scaled)
    np.testing.assert_allclose(original[MODEL_FEATURES],other[MODEL_FEATURES],equal_nan=True,atol=1e-12)
    assert not {'target_return','entry_price','exit_price','symbol','date'}&set(MODEL_FEATURES)


def test_boundary_purge_and_tied_relevance():
    dates=pd.bdate_range('2020-12-01','2022-02-01')
    frame=pd.DataFrame({'date':dates,'symbol':'AAA','model_ready':True,'target_return':.01,
                        'label_end_date':pd.Series(dates).shift(-6)})
    frame.loc[frame.label_end_date.isna(),'target_return']=np.nan
    train,validation,test=chronological_splits(frame)
    assert train.label_end_date.max()<validation.date.min()
    assert validation.label_end_date.max()<test.date.min()
    assert test.target_return.isna().any()  # Inference retains rows without known outcomes.
    query=pd.DataFrame({'date':[dates[0]]*3,'symbol':['AAA','BBB','CCC'],'target_return':[.1,.1,.2],**{name:1. for name in MODEL_FEATURES}})
    result,y,groups=ranking_query(query)
    assert y.iloc[0]==y.iloc[1] and y.iloc[2]>y.iloc[0]
    assert groups.tolist()==[3]


def test_training_save_reload_and_missing_outcome_ranking(tmp_path):
    rng=np.random.default_rng(42);n=240
    frame=pd.DataFrame(rng.normal(size=(n,len(MODEL_FEATURES))),columns=MODEL_FEATURES)
    frame['date']=np.repeat(pd.bdate_range('2020-01-01',periods=20),12)
    frame['symbol']=np.tile([f'S{i:02}' for i in range(12)],20)
    frame['target_return']=frame.ema_9_distance*.01
    query,y,groups=ranking_query(frame)
    model=new_ranker(5);model.fit(query[MODEL_FEATURES],y,group=groups)
    model.booster_.save_model(str(tmp_path/'model.txt'))
    loaded=lgb.Booster(model_file=str(tmp_path/'model.txt'))
    expected=score_frame(model,frame);actual=score_frame(loaded,frame)
    np.testing.assert_allclose(expected.score,actual.score,rtol=1e-12)
    changed=frame.copy();changed['target_return']=np.nan
    pd.testing.assert_frame_equal(expected[['date','symbol','score','rank']],score_frame(loaded,changed)[['date','symbol','score','rank']])
    metrics,_=evaluate(expected)
    assert metrics['evaluated_sessions']==20 and metrics['complete_top_10_sessions']==20
    with pytest.raises(ValueError,match='outcomes'):evaluate(score_frame(loaded,changed))


def test_complete_training_and_inference_artifacts(tmp_path,monkeypatch):
    import json
    from models import train_swing_ranker as trainer
    from models.predict_swing import predict
    stock,dates=prices(1100)
    saved=calculate_features(stock,pd.DataFrame({'date':dates,'nifty500_close':1000.}))
    prepared=model_inputs(saved).merge(execution_labels(stock,dates),on=['date','symbol'],validate='one_to_one')
    monkeypatch.setattr(trainer,'load_dataset',lambda *args:(prepared,[]))
    report=trainer.train(output=tmp_path/'model',max_estimators=5,patience=2)
    assert pd.Timestamp(report['final_training']['latest_label_end'])<pd.Timestamp(report['test']['first_signal'])
    assert report['live_trading_enabled'] is False
    assert (tmp_path/'model/ranker.txt').is_file()
    stored=pd.read_csv(tmp_path/'model/test_predictions.csv.xz')
    assert stored.target_return.isna().any()
    assert pd.to_datetime(stored.date).dt.year.eq(2022).all()
    assert json.loads((tmp_path/'model/report.json').read_text())['model_sha256']==report['model_sha256']
    # Changing held-out outcomes must not change selected trees or fitted model.
    prepared.loc[prepared.date.dt.year.eq(2022),'target_return'] *= -1
    alternate=trainer.train(output=tmp_path/'alternate',max_estimators=5,patience=2)
    assert alternate['model_sha256']==report['model_sha256']
    assert alternate['best_iteration']==report['best_iteration']
    path=tmp_path/'features.csv.xz';saved.to_csv(path,index=False)
    result=predict(path,tmp_path/'model',date='2022-06-01',top_k=2)
    assert len(result)==2 and result['rank'].tolist()==[1,2]
    with pytest.raises(ValueError,match='training outcomes'):
        predict(path,tmp_path/'model',date='2021-06-01')
    with pytest.raises(ValueError,match='absent'):
        predict(path,tmp_path/'model',date='2030-01-01')
