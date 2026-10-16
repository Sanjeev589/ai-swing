"""Synthetic tests for a paper-only swing research system; no profitability claims."""
import numpy as np
import pandas as pd
import pytest

from backtesting.ai_swing import SwingResearchConfig, simulate, prepare_candidates, purged_predictions


def config(**kwargs):
    return SwingResearchConfig(test_start='2022-01-03', test_end='2022-01-14',
                               min_train_sessions=5, max_positions=1,
                               min_turnover_inr=0, min_breadth_stocks=1, **kwargs)


def fixture_bars():
    dates = pd.bdate_range('2022-01-03', periods=10)
    bars = pd.DataFrame({'date': dates, 'symbol': 'AAA',
                         'open': [100., 101., 94., 93., 100., 100., 100., 100., 100., 100.],
                         'high': [101., 103., 96., 95., 102., 102., 102., 102., 102., 102.],
                         'low': [99., 98., 93., 91., 98., 98., 98., 98., 98., 98.],
                         'close': [100., 100., 95., 94., 100., 100., 100., 100., 100., 100.],
                         'volume': 100000})
    return bars


def signal(on):
    return pd.DataFrame([{'date': pd.Timestamp(on), 'symbol': 'AAA', 'setup': 'breakout',
                          'ai_score': 1., 'rule_score': .1, 'atr_14': 2., 'close': 100.}])


def test_gap_stop_executes_at_actual_open_not_stop():
    cfg = config(cost_bps_per_side=0)
    equity, trades, stats = simulate(fixture_bars(), signal('2022-01-03'), cfg)
    assert trades.iloc[0].entry_date == pd.Timestamp('2022-01-04')
    assert trades.iloc[0].exit_date == pd.Timestamp('2022-01-05')
    assert trades.iloc[0].exit_reason == 'gap_stop'
    assert trades.iloc[0].exit_price == 94  # worse than stop at 97
    assert trades.iloc[0].net_pnl_inr < 0
    assert stats['paper_only'] and stats['open_positions'] == 0
    assert abs(equity.iloc[-1].equity - (cfg.initial_capital + trades.net_pnl_inr.sum())) < 1e-5


def test_no_same_day_entry_and_never_buy_final_signal():
    prices = fixture_bars()
    sigs = pd.concat([signal('2022-01-14'), signal('2022-01-03')], ignore_index=True)
    _, trades, _ = simulate(prices, sigs, config(cost_bps_per_side=0))
    assert len(trades) == 1
    assert trades.entry_date.min() > pd.Timestamp('2022-01-03')
    assert not trades.signal_date.eq(pd.Timestamp('2022-01-14')).any()


def test_stop_updates_only_after_day_close():
    prices = fixture_bars()
    # At entry the initial stop is 97. Intraday low 97, stops out. Set low 98:
    prices.loc[prices.date.eq('2022-01-04'), 'low'] = 98.
    prices.loc[prices.date.eq('2022-01-04'), 'close'] = 105.
    prices.loc[prices.date.eq('2022-01-04'), 'high'] = 106.
    # Trailing stop raises to 101 at close; opening next day at 100 fills at 100.
    prices.loc[prices.date.eq('2022-01-05'), 'open'] = 100.
    _, trades, _ = simulate(prices, signal('2022-01-03'), config(cost_bps_per_side=0))
    assert trades.iloc[0].exit_date == pd.Timestamp('2022-01-05')
    assert trades.iloc[0].exit_price == 100
    assert trades.iloc[0].exit_reason == 'gap_stop'


def test_missing_position_prices_fail_closed():
    prices = fixture_bars()
    prices = prices[prices.date.ne(pd.Timestamp('2022-01-05'))]
    # Include another stock so the exchange stays open on 5 Jan.
    other = fixture_bars().assign(symbol='BBB')
    with pytest.raises(ValueError, match='Unpriced open positions'):
        simulate(pd.concat([prices, other]), signal('2022-01-03'), config())


def test_invalid_price_inputs_and_risk_rejection(tmp_path):
    from backtesting.ai_swing import load_ohlcv, run
    with pytest.raises(ValueError, match='positive integer'):
        SwingResearchConfig(horizon=0)
    with pytest.raises(ValueError, match='Explicit --price-policy'):
        run('not_needed', 'not_needed', tmp_path, price_policy=None)
    incomplete = fixture_bars().drop(columns=['high', 'low'])
    csv = tmp_path / 'incomplete.csv'
    incomplete.to_csv(csv, index=False)
    with pytest.raises(ValueError, match='high|low'):
        load_ohlcv(csv)


def test_empty_signals_remain_in_cash():
    equity, trades, stats = simulate(fixture_bars(), pd.DataFrame(columns=['date', 'symbol', 'ai_score']),
                                     config(cost_bps_per_side=0))
    assert trades.empty
    assert equity.equity.eq(1_000_000).all()
    assert stats['win_rate'] is None and stats['total_net_return'] == 0


def test_purged_training_scores_unchanged_by_test_labels():
    from models.swing_dataset import MODEL_FEATURES
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2021-11-01', periods=80)
    rows = []
    for i, date in enumerate(dates):
        for j in range(8):
            row = {'date': date, 'symbol': f'S{j}',
                   'model_ready': True, 'eligible': True,
                   'label_end_date': dates[i+6] if i+6 < len(dates) else pd.NaT,
                   'target_return': rng.normal() if i+6 < len(dates) else np.nan,
                   'relative_strength_20': rng.normal(), 'relative_strength_60': rng.normal(),
                   'breakout': j % 2 == 0}
            row.update({name: rng.normal() for name in MODEL_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)
    cfg = SwingResearchConfig(test_start='2022-01-19', test_end='2022-02-18',
                              min_train_sessions=25, refit_sessions=8, estimators=5)
    first, folds = purged_predictions(frame, cfg)
    assert folds.last_training_label.lt(folds.test_start).all()
    assert (folds.train_sessions >= 25).all()
    changed = frame.copy()
    changed.loc[changed.date.ge('2022-01-19'), 'target_return'] = -200
    again, _ = purged_predictions(changed, cfg)
    # Later folds may legitimately retrain on earlier TEST outcomes once their
    # label_end_date precedes the next fold. Only the initial fold must be invariant.
    original = first[first.fold.eq(0)]
    revised = again[again.fold.eq(0)]
    np.testing.assert_allclose(original.ai_score, revised.ai_score)