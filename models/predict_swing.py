"""Rank one saved-feature session using the trained native LightGBM model."""
import argparse
import json
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from models.swing_dataset import MODEL_FEATURES,model_inputs
from models.train_swing_ranker import score_frame


def predict(feature_file,model_dir='models/trained/swing_ranker',date=None,top_k=10,output=None):
    if top_k<1: raise ValueError('top_k must be positive')
    model_dir=Path(model_dir)
    report=json.loads((model_dir/'report.json').read_text())
    if report['features']!=MODEL_FEATURES: raise ValueError('Model feature schema mismatch')
    frame=model_inputs(pd.read_csv(feature_file,parse_dates=['date']))
    session=pd.Timestamp(date) if date else frame.date.max()
    frame=frame[frame.date.eq(session)]
    if frame.empty: raise ValueError('Requested session is absent')
    trained_through=pd.Timestamp(report['final_training']['latest_label_end'])
    if session<=trained_through: raise ValueError('Requested date overlaps model training outcomes')
    model=lgb.Booster(model_file=str(model_dir/'ranker.txt'))
    if model.feature_name()!=MODEL_FEATURES: raise ValueError('Model feature order mismatch')
    ranked=score_frame(model,frame).head(top_k)
    if ranked.empty: raise ValueError('No stocks have complete model features')
    result=ranked[['date','symbol','score','rank']]
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path,index=False,date_format='%Y-%m-%d')
    print(result.to_string(index=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-file',required=True);parser.add_argument('--model-dir',default='models/trained/swing_ranker')
    parser.add_argument('--date');parser.add_argument('--top-k',type=int,default=10);parser.add_argument('--output')
    predict(**vars(parser.parse_args()))
