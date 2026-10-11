import pandas as pd
import pytest
from data.nse_loader import load_daily_csv
from features.technical_indicators import build_features
from backtesting.backtest import net_long_return
from execution.broker import Order, PaperBroker

def test_loader_and_features(tmp_path):
    path = tmp_path / "prices.csv"
    rows = []
    for symbol, offset in (("AAA", 0), ("BBB", 50)):
        for i, date in enumerate(pd.bdate_range("2024-01-01", periods=80)):
            price = 100 + offset + i
            rows.append(dict(date=date, symbol=symbol, open=price, high=price+1,
                             low=price-1, close=price, volume=1000+i))
    pd.DataFrame(rows).to_csv(path, index=False)
    data = load_daily_csv(path)
    features = build_features(data)
    assert len(features) == 160
    assert features.loc[features.symbol == "AAA", "return_5"].iloc[0] != features.loc[features.symbol == "AAA", "return_5"].iloc[0]
    assert features.loc[features.symbol == "BBB", "target_5d_return"].iloc[-1] != features.loc[features.symbol == "BBB", "target_5d_return"].iloc[-1]

def test_paper_only():
    broker = PaperBroker()
    assert broker.place_order(Order("RELIANCE", 1, "BUY"))["status"] == "PAPER_ONLY"
    with pytest.raises(ValueError):
        broker.place_order(Order("RELIANCE", -1, "BUY"))

def test_cost():
    assert net_long_return(100, 110, .002) == pytest.approx(.098)
