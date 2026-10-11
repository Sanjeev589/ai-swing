import numpy as np
import pandas as pd
import pytest
from data.nse_loader import validate_daily
from data.historical import download_nse
from features.technical_indicators import FEATURES, TARGET, build_features
from backtesting.walk_forward import (WalkForwardConfig, walk_forward_predictions,
                                      portfolio_backtest, run_walk_forward)
from models.ranking import fit_ranker, rank_stocks


@pytest.fixture
def prices():
    rng = np.random.default_rng(10)
    rows = []
    for symbol in ('AAA', 'BBB', 'CCC'):
        close = 100 * np.exp(np.cumsum(rng.normal(.001, .01, 180)))
        for i, date in enumerate(pd.bdate_range('2023-01-01', periods=180)):
            opening = close[i] * 1.002
            rows.append(dict(date=date, symbol=symbol, open=opening,
                             high=opening * 1.02, low=close[i] * .98,
                             close=close[i], volume=1000+i))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('column,value', [('close', np.nan), ('volume', np.inf),
                                         ('symbol', None), ('date', None), ('symbol', ' ')])
def test_invalid_data(prices, column, value):
    if column == "volume":
        prices[column] = prices[column].astype(float)
    prices.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_daily(prices)


def test_labels_and_feature_causality(prices):
    original = build_features(prices, 5)
    stock = original[original.symbol.eq('AAA')].reset_index(drop=True)
    assert stock.loc[60, TARGET] == pytest.approx(stock.loc[66, 'open'] / stock.loc[61, 'open'] - 1)
    assert stock.loc[60, 'label_end_date'] == stock.loc[66, 'date']
    assert stock[TARGET].tail(6).isna().all()
    cutoff = sorted(prices.date.unique())[100]
    changed = prices.copy()
    changed.loc[changed.date > cutoff, ['open', 'high', 'low', 'close']] *= 2
    rebuilt = build_features(changed, 5)
    pd.testing.assert_frame_equal(original.loc[original.date <= cutoff, FEATURES],
                                  rebuilt.loc[rebuilt.date <= cutoff, FEATURES])


def test_walk_forward_purge_and_future_invariance(prices):
    config = WalkForwardConfig(min_train_sessions=30, test_sessions=20, n_estimators=10)
    predicted, folds = walk_forward_predictions(build_features(prices), config)
    assert (folds.latest_training_label < folds.test_start).all()
    assert (folds.train_end < folds.test_start).all()
    assert not predicted.duplicated(['date', 'symbol']).any()
    boundary = folds.iloc[0].test_end
    changed = prices.copy()
    changed.loc[changed.date > boundary, ['open', 'high', 'low', 'close']] *= 3
    newer, _ = walk_forward_predictions(build_features(changed), config)
    pd.testing.assert_frame_equal(predicted.loc[predicted.date <= boundary, ['date', 'symbol', 'score', 'rank']].reset_index(drop=True),
                                  newer.loc[newer.date <= boundary, ['date', 'symbol', 'score', 'rank']].reset_index(drop=True))


def test_ranking_deterministic(prices):
    features = build_features(prices)
    model = fit_ranker(features, n_estimators=10)
    ranked = rank_stocks(model, features)
    assert ranked.groupby('date')['rank'].apply(list).apply(lambda x: x == [1, 2, 3]).all()
    assert np.isfinite(ranked.score).all()
    assert set(FEATURES) == set(model.feature_name_)


def test_portfolio_exact_cost_cash_and_nonoverlap():
    dates = pd.bdate_range('2024-01-01', periods=12)
    rows = [dict(date=d, symbol='AAA', open=100., high=100., low=100., close=100., volume=1000)
            for d in dates]
    predictions = pd.DataFrame({'date': dates, 'symbol': 'AAA', 'score': 1.})
    config = WalkForwardConfig(horizon=5, top_k=1, max_weight=.5, cost_bps_per_side=10)
    curve, trades, summary = portfolio_backtest(pd.DataFrame(rows), predictions, config)
    factor = (1-.001)/(1+.001)
    expected = (.5 + .5 * factor) ** 2
    assert summary['total_return'] == pytest.approx(expected - 1)
    assert len(trades) == 2
    assert trades.iloc[0].exit_date == trades.iloc[1].entry_date
    assert (curve.cash >= -1e-12).all()
    assert curve.iloc[-1].positions == 0
    assert trades.weight.eq(.5).all()
    assert summary['max_drawdown'] == pytest.approx(expected - 1)


def test_end_to_end_and_missing_panel(prices, tmp_path):
    path = tmp_path / 'daily.csv'
    prices.to_csv(path, index=False)
    config = WalkForwardConfig(min_train_sessions=30, test_sessions=20, n_estimators=10)
    result = run_walk_forward(path, config)
    assert result['summary']['completed_trades'] > 0
    assert result['summary']['live_trading'] is False
    with pytest.raises(ValueError, match='complete session panel'):
        portfolio_backtest(prices.iloc[1:], result['predictions'], config)


def test_download_cache_and_failure(tmp_path):
    calls = []
    def downloader(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return pd.DataFrame({'Open': [100], 'High': [102], 'Low': [99],
                             'Close': [101], 'Volume': [1000]},
                            index=pd.DatetimeIndex(['2024-01-02'], tz='Asia/Kolkata'))
    output = tmp_path / 'daily.csv'
    result = download_nse(['reliance', 'TCS.NS'], '2024-01-01', '2024-01-05', output, downloader=downloader)
    assert result.symbol.tolist() == ['RELIANCE', 'TCS']
    assert calls[0][0] == 'RELIANCE.NS'
    assert calls[0][1]['auto_adjust'] is True
    assert output.with_suffix('.csv.metadata.json').exists()
    before = output.read_text()
    with pytest.raises(ValueError, match='No historical data'):
        download_nse(['BAD'], '2024-01-01', '2024-01-05', output, downloader=lambda *a, **k: pd.DataFrame())
    assert output.read_text() == before


@pytest.mark.parametrize('kwargs', [{'horizon': 0}, {'top_k': 1.5}, {'max_weight': 0},
                                   {'cost_bps_per_side': np.nan}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        WalkForwardConfig(**kwargs)


def test_regression_baseline_and_invalid_orders(prices, tmp_path):
    from models.train_lightgbm import fit_and_evaluate
    from execution.broker import PaperBroker, Order
    from backtesting.backtest import net_long_return
    path = tmp_path / 'daily.csv'
    prices.to_csv(path, index=False)
    dates = sorted(prices.date.unique())
    _, output = fit_and_evaluate(path, str(pd.Timestamp(dates[90]).date()),
                                str(pd.Timestamp(dates[135]).date()))
    assert len(output) > 0
    broker = PaperBroker()
    for quantity in (1.5, True, 0):
        with pytest.raises(ValueError):
            broker.place_order(Order('AAA', quantity, 'BUY'))
    with pytest.raises(ValueError):
        net_long_return(float('nan'), 100)
