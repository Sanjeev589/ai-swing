"""Model inputs from saved indicators; daily-calendar execution labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from data.feature_inputs import validate_prices

MODEL_FEATURES = [
    'ema_9_distance', 'ema_21_distance', 'ema_50_distance', 'rsi_14',
    'macd_fraction', 'macd_signal_fraction', 'macd_histogram_fraction',
    'volume_ratio_20', 'adx_14', 'plus_di_14', 'minus_di_14',
    'bb_position', 'bb_width', 'atr_14_fraction',
    'supertrend_distance', 'supertrend_direction',
    'relative_strength_20', 'relative_strength_60', 'high_252_proximity',
]
REQUIRED = ['date','symbol','close','ema_9','ema_21','ema_50','rsi_14',
            'macd','macd_signal','macd_histogram','volume_ratio_20','adx_14',
            'plus_di_14','minus_di_14','bb_middle_20','bb_upper_20_2','bb_lower_20_2',
            'bb_width','atr_14','supertrend_10_3','supertrend_direction',
            'relative_strength_20','relative_strength_60','high_252_proximity']


def model_inputs(frame):
    if not set(REQUIRED).issubset(frame.columns):
        raise ValueError('Missing saved indicator columns')
    result = frame.copy()
    result['date'] = pd.to_datetime(result.date, errors='raise')
    if result[['date','symbol']].isna().any().any() or result.duplicated(['date','symbol']).any():
        raise ValueError('Invalid feature keys')
    for name in REQUIRED[2:]:
        result[name] = pd.to_numeric(result[name], errors='raise')
    if result.close.le(0).any() or not np.isfinite(result.close).all():
        raise ValueError('Invalid feature closes')
    for period in [9,21,50]:
        result[f'ema_{period}_distance'] = result.close/result[f'ema_{period}']-1
    for name in ['macd','macd_signal','macd_histogram']:
        result[f'{name}_fraction'] = result[name]/result.close
    width = result.bb_upper_20_2-result.bb_lower_20_2
    result['bb_position'] = ((result.close-result.bb_lower_20_2)/width.replace(0,np.nan)).mask(width.eq(0),.5)
    result['atr_14_fraction'] = result.atr_14/result.close
    result['supertrend_distance'] = result.close/result.supertrend_10_3-1
    result[MODEL_FEATURES] = result[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan)
    result['model_ready'] = result[MODEL_FEATURES].notna().all(axis=1)
    return result.sort_values(['date','symbol']).reset_index(drop=True)


def execution_labels(prices, calendar, horizon=5):
    if isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<1:
        raise ValueError('Horizon must be a positive integer')
    prices = validate_prices(prices)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    if calendar.has_duplicates or calendar.isna().any() or calendar.tz is not None or not calendar.equals(calendar.normalize()):
        raise ValueError('Invalid daily benchmark calendar')
    if not prices.date.isin(calendar).all():
        raise ValueError('Price date missing from benchmark calendar')
    results = []
    for symbol, stock in prices.groupby('symbol',sort=True):
        bars = stock.set_index('date').reindex(calendar)
        valid = bars[['open','high','low','close']].gt(0).all(axis=1) & bars.volume.gt(0)
        valid &= bars.high.ge(bars[['open','low','close']].max(axis=1)) & bars.low.le(bars[['open','high','close']].min(axis=1))
        complete = valid.astype(int).rolling(horizon+1,min_periods=horizon+1).sum().shift(-(horizon+1)).eq(horizon+1)
        labels = pd.DataFrame({'date':calendar,'symbol':symbol})
        labels['entry_date'] = pd.Series(calendar).shift(-1)
        labels['label_end_date'] = pd.Series(calendar).shift(-(horizon+1))
        labels['entry_price'] = bars.open.shift(-1).to_numpy()
        labels['exit_price'] = bars.open.shift(-(horizon+1)).to_numpy()
        labels['target_return'] = np.where(complete,labels.exit_price/labels.entry_price-1,np.nan)
        results.append(labels)
    return pd.concat(results,ignore_index=True)


def load_dataset(features='features/derived', inputs='data/feature_inputs', horizon=5):
    features, inputs = Path(features),Path(inputs)
    paths = sorted(features.glob('features_*.csv.xz'))
    price_paths = sorted(inputs.glob('ohlcv_*.csv.gz'))
    if not paths or not price_paths: raise ValueError('Missing feature or price files')
    frame = model_inputs(pd.concat([pd.read_csv(p,parse_dates=['date']) for p in paths],ignore_index=True))
    prices = validate_prices(pd.concat([pd.read_csv(p,parse_dates=['date']) for p in price_paths],ignore_index=True))
    check = frame[['date','symbol','close']].merge(prices[['date','symbol','close']],on=['date','symbol'],how='left',suffixes=('_feature','_source'),validate='one_to_one')
    if check.close_source.isna().any() or not np.allclose(check.close_feature,check.close_source,atol=1e-8,rtol=0):
        raise ValueError('Saved features do not match source closes')
    benchmark = pd.read_csv(inputs/'nifty500_index.csv',parse_dates=['date'])
    labels = execution_labels(prices,benchmark.date,horizon)
    return frame.merge(labels,on=['date','symbol'],how='left',validate='one_to_one'),paths+price_paths+[inputs/'nifty500_index.csv']


def chronological_splits(frame,train_end='2020-12-31',validation_end='2021-12-31',test_end='2022-12-31'):
    train_end,validation_end,test_end = map(pd.Timestamp,[train_end,validation_end,test_end])
    if not train_end<validation_end<test_end: raise ValueError('Split dates must increase')
    eligible = frame[frame.model_ready & frame.target_return.notna()].copy()
    train = eligible[(eligible.date<=train_end)&(eligible.label_end_date<=train_end)]
    validation = eligible[(eligible.date>train_end)&(eligible.date<=validation_end)&(eligible.label_end_date<=validation_end)]
    test = frame[frame.model_ready & frame.date.gt(validation_end)&frame.date.le(test_end)].copy()
    for name,part in [('train',train),('validation',validation),('test',test)]:
        if part.empty: raise ValueError(f'Empty {name} split')
    if train.label_end_date.max()>=validation.date.min() or validation.label_end_date.max()>=test.date.min():
        raise ValueError('Labels cross split boundaries')
    return train,validation,test
