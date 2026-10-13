"""Train and persist a chronological LightGBM ranking research model."""
import argparse
import hashlib
import json
import lzma
import platform
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from models.swing_dataset import MODEL_FEATURES,load_dataset,chronological_splits


def ranking_query(frame):
    result = frame.dropna(subset=MODEL_FEATURES+['target_return']).sort_values(['date','symbol']).copy()
    result = result[result.groupby('date').symbol.transform('size').ge(2)]
    if result.empty: raise ValueError('Ranking requires at least two stocks per session')
    percentile = result.groupby('date').target_return.rank(method='average',pct=True)
    relevance = np.minimum((percentile*5).astype(int),4)
    return result,relevance,result.groupby('date',sort=True).size().to_numpy()


def new_ranker(iterations,seed=42):
    return lgb.LGBMRanker(objective='lambdarank',label_gain=[0,1,2,3,4],
        n_estimators=iterations,learning_rate=.03,num_leaves=15,min_child_samples=100,
        reg_lambda=1.,random_state=seed,n_jobs=2,verbosity=-1,
        deterministic=True,force_col_wise=True,lambdarank_truncation_level=20)


def score_frame(model,frame):
    result = frame[frame[MODEL_FEATURES].notna().all(axis=1)].copy()
    if result.empty:
        return result.assign(score=np.nan, rank=pd.Series(dtype='int64'))
    result['score'] = model.predict(result[MODEL_FEATURES],validate_features=True,num_threads=2)
    return rank_scores(result)


def rank_scores(frame):
    result = frame.sort_values(['date','score','symbol'],ascending=[True,False,True]).copy()
    result['rank'] = result.groupby('date').cumcount()+1
    return result


def evaluate(predictions,top_k=10):
    daily = []
    for date,group in predictions.groupby('date',sort=True):
        observed = group.dropna(subset=['target_return'])
        if len(observed)<2: continue
        ranks = observed.target_return.rank(method='average',pct=True)
        relevance = np.minimum((ranks*5).astype(int),4)
        ic = observed.score.rank().corr(observed.target_return.rank()) if observed.score.nunique()>1 and observed.target_return.nunique()>1 else np.nan
        selected = group[group['rank']<=top_k]
        complete = len(selected)==top_k and selected.target_return.notna().all()
        daily.append({'date':date,'eligible_stocks':len(group),'known_outcomes':len(observed),
            'rank_ic':ic,'ndcg_at_10':ndcg_score([relevance.to_numpy()],[observed.score.to_numpy()],k=top_k),
            'top_10_gross_return':float(selected.target_return.mean()) if complete else np.nan,
            'universe_gross_return':float(observed.target_return.mean()),
            'complete_top_10':complete})
    daily = pd.DataFrame(daily)
    if daily.empty: raise ValueError('No known evaluation outcomes')
    comparable = daily[daily.complete_top_10]
    def value(series):
        mean = series.mean()
        return float(mean) if pd.notna(mean) else None
    metrics = {'signal_sessions':predictions.date.nunique(),'evaluated_sessions':len(daily),
        'prediction_rows':len(predictions),'labelled_rows':int(predictions.target_return.notna().sum()),
        'mean_daily_rank_ic':value(daily.rank_ic),'mean_daily_ndcg_at_10':value(daily.ndcg_at_10),
        'complete_top_10_sessions':len(comparable),'mean_top_10_gross_return':value(comparable.top_10_gross_return),
        'mean_universe_gross_return_same_sessions':value(comparable.universe_gross_return),
        'mean_top_10_excess_return':value(comparable.top_10_gross_return-comparable.universe_gross_return),
        'label_abs_return_over_50pct':int(predictions.target_return.abs().gt(.5).sum())}
    return metrics,daily


def train(features='features/derived',inputs='data/feature_inputs',output='models/trained/swing_ranker',
          horizon=5,max_estimators=300,patience=30,seed=42,train_end='2020-12-31',
          validation_end='2021-12-31',test_end='2022-12-31'):
    if max_estimators<1 or patience<1: raise ValueError('Invalid training iterations')
    frame,paths = load_dataset(features,inputs,horizon)
    train_frame,validation,test = chronological_splits(frame,train_end,validation_end,test_end)
    train_frame,y,groups = ranking_query(train_frame)
    validation,vy,vgroups = ranking_query(validation)
    model = new_ranker(max_estimators,seed)
    model.fit(train_frame[MODEL_FEATURES],y,group=groups,
        eval_set=[(validation[MODEL_FEATURES],vy)],eval_group=[vgroups],eval_at=[10],
        callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False)])
    best = model.best_iteration_ or max_estimators
    validation_predictions = score_frame(model,validation)
    validation_metrics,_ = evaluate(validation_predictions)
    # Refit on train+validation only; every outcome predates the held-out test.
    final_frame = pd.concat([train_frame,validation],ignore_index=True)
    final_frame,fy,fgroups = ranking_query(final_frame)
    if final_frame.label_end_date.max()>=test.date.min(): raise ValueError('Test leakage')
    final_model = new_ranker(best,seed)
    final_model.fit(final_frame[MODEL_FEATURES],fy,group=fgroups)
    predictions = score_frame(final_model,test)
    predictions.loc[predictions.label_end_date.gt(pd.Timestamp(test_end)),'target_return'] = np.nan
    test_metrics,daily = evaluate(predictions)
    baseline = rank_scores(predictions.assign(score=predictions.relative_strength_20))
    baseline_metrics,_ = evaluate(baseline)
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    final_model.booster_.save_model(str(output/'ranker.txt'))
    # Native text model round-trip must preserve predictions and feature order.
    loaded = lgb.Booster(model_file=str(output/'ranker.txt'))
    assert loaded.feature_name()==MODEL_FEATURES
    np.testing.assert_allclose(loaded.predict(predictions[MODEL_FEATURES],num_threads=2),predictions.score,rtol=1e-12,atol=1e-12)
    columns = ['date','symbol','score','rank','entry_date','label_end_date','entry_price','exit_price','target_return']
    (output/'test_predictions.csv.xz').write_bytes(lzma.compress(predictions[columns].to_csv(index=False,date_format='%Y-%m-%d',float_format='%.10g').encode()))
    daily.to_csv(output/'test_daily_metrics.csv',index=False,date_format='%Y-%m-%d')
    importance = pd.DataFrame({'feature':MODEL_FEATURES,'gain':loaded.feature_importance(importance_type='gain'),'splits':loaded.feature_importance(importance_type='split')})
    importance.sort_values('gain',ascending=False).to_csv(output/'feature_importance.csv',index=False)
    def coverage(part):
        return {'rows':len(part),'sessions':part.date.nunique(),'first_signal':str(part.date.min().date()),
                'last_signal':str(part.date.max().date()),'latest_label_end':str(part.label_end_date.dropna().max().date())}
    report = {'model':'LightGBM LambdaRank','features':MODEL_FEATURES,'horizon_open_to_open_sessions':horizon,
        'split_boundaries':{'train_end':train_end,'validation_end':validation_end,'test_end':test_end},
        'selection_train':coverage(train_frame),'validation':coverage(validation),'final_training':coverage(final_frame),
        'test':coverage(test),'best_iteration':best,'validation_metrics':validation_metrics,
        'test_metrics':test_metrics,'relative_strength_20_baseline':baseline_metrics,
        'parameters':final_model.get_params(),'versions':{'python':platform.python_version(),'lightgbm':lgb.__version__,'pandas':pd.__version__,'numpy':np.__version__},
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'model_sha256':hashlib.sha256((output/'ranker.txt').read_bytes()).hexdigest(),
        'live_trading_enabled':False,
        'limitations':['Unadjusted prices: corporate actions can create misleading features and labels.',
                       'Current constituent snapshot: historical membership and delistings are not reconstructed.',
                       'Ranking scores are not returns or probabilities.',
                       'Metrics use overlapping five-session gross outcomes; no fees, portfolio CAGR, or live-performance claim.',
                       'NDCG and IC use available outcome rows; top-10 mean requires all ten outcomes observed.',
                       '2021 selects tree count; final model trains only on selected 2019–2021 rows; 2022 is held out.']}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'best_iteration':best,'test':test_metrics,'baseline':baseline_metrics},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',default='features/derived');parser.add_argument('--inputs',default='data/feature_inputs')
    parser.add_argument('--output',default='models/trained/swing_ranker');parser.add_argument('--horizon',type=int,default=5)
    parser.add_argument('--max-estimators',type=int,default=300);parser.add_argument('--patience',type=int,default=30)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-end',default='2020-12-31');parser.add_argument('--validation-end',default='2021-12-31');parser.add_argument('--test-end',default='2022-12-31')
    train(**vars(parser.parse_args()))
