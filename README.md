# India AI Swing Trading

Research-first Python project for daily NSE equity swing-trading signals. **No live orders** are placed.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Supply your own legally obtained daily OHLCV CSV under `data/local/` with columns `date,symbol,open,high,low,close,volume`. Prices should be adjusted consistently for corporate actions. Do not commit proprietary data or secrets.

## Pipeline
1. `data/nse_loader.py`: validate and load daily OHLCV CSV
2. `features/technical_indicators.py`: compute lag-safe per-symbol features
3. `models/train_lightgbm.py`: chronological train/validation/test model; predict 5-day close-to-close returns
4. `backtesting/backtest.py`: simple next-open entry and 5-day hold research baseline
5. `agents/`: reserved for research and monitoring agents, **not** discretionary live execution
6. `execution/broker.py`: paper-only broker contract

## Example
```powershell
python -m models.train_lightgbm --csv data/local/daily.csv
```

This is an educational research baseline, not an investable strategy. The target is close-to-close, whereas the illustrative backtest enters at the next open; refine label/execution alignment, survivorship bias, point-in-time universe, corporate actions, realistic NSE fees/slippage, walk-forward validation, and market-regime tests before paper trading.
