# India AI Swing Trading

Python research pipeline for daily NSE equity swing trading. **Live trading is disabled.** The only broker is `PaperBroker`; no broker credentials or live-order code are used.

## Windows / VS Code setup

Open the repository folder in VS Code, install its Python extension, and select the `.venv` interpreter. The checked-in VS Code settings enable pytest discovery.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The commands below assume the virtual environment is selected/activated; alternatively use its full Python path. Python 3.11 and 3.12 are covered by the GitHub Actions Windows/Linux test matrix.

## 1. Historical NSE-symbol data

Download third-party research data for NSE symbols via Yahoo Finance's `.NS` tickers:

```powershell
python -m data.historical --symbols RELIANCE TCS INFY HDFCBANK ICICIBANK --start 2018-01-01 --end 2026-10-01 --output data/local/daily.csv
```

Start is inclusive; end is exclusive. The adapter explicitly requests adjusted daily OHLC, validates every symbol, and writes a CSV cache plus a provenance metadata JSON. Empty/invalid downloads fail without replacing an existing CSV. Downloading is an explicit step: training and backtesting operate offline on cached CSVs. Provider rate limits may block a download; retry later or supply your own CSV. Yahoo is not an official NSE feed and its history may contain errors or revisions; check source permissions and reconcile corporate actions before drawing conclusions.

Alternatively supply a legally obtained daily CSV with `date,symbol,open,high,low,close,volume`. Dates must be timezone-naive session dates, prices positive and consistently adjusted, and volume finite/nonnegative. Empty symbols, invalid OHLC ranges, missing fields, and duplicate symbol/date rows are rejected. Data and generated artifacts are git-ignored.

## 2. Features and stock ranking

`features/technical_indicators.py` computes symbol-isolated momentum, moving-average distances, volatility, volume ratio, RSI, and ATR. Each feature is available at that session's close. Warm-up rows remain missing rather than being filled from future observations.

A signal at close on session **t** enters at open **t+1** and exits at open **t+1+horizon**. The default horizon is five open-to-open session intervals. `target_return`, entry/exit prices, and `label_end_date` are calculated on the original symbol calendar before any rows are dropped. The legacy `target_5d_return` alias now uses this execution-aligned definition even when a different horizon is requested.

`models/ranking.py` uses LightGBM LambdaRank, with each date's cross-section as a query and five relevance levels derived from forward-return ranks. Tied returns have equal relevance. Scores are **ranking scores, not predicted returns**. Symbol order breaks equal prediction scores deterministically. Feature columns explicitly exclude labels, dates, and future prices.

## 3. Purged walk-forward backtesting

```powershell
python -m backtesting.walk_forward --csv data/local/daily.csv --min-train-sessions 252 --test-sessions 63 --horizon 5 --top-k 10 --max-weight 0.10 --cost-bps-per-side 10 --output artifacts/walk_forward
```

Training expands through time. Each fold requires 252 eligible training sessions (in addition to the 50-session feature warm-up and label purge). A training label's exit date must strictly precede the first test signal date. Hyperparameters are fixed; test outcomes are never used for fitting or tuning. Models refit every 63 signal sessions, including partial final folds.

The portfolio selects top-ranked stocks every five sessions and buys at the next open. Positions exit before the next batch enters, so cohorts do not overlap. Each stock receives at most the configured maximum weight; unused allocation stays in cash. Fractional research shares permit normalized starting equity of 1.0. Costs are charged on entry and exit; the default 10 bps per side is an illustrative combined fee/slippage proxy, **not an exact NSE tax schedule**.

Equity is marked at each session open, including entry/exit costs and cash. Incomplete final trades are skipped. The fixed-universe backtest requires all symbols to share a complete price calendar: it refuses to forward-fill missing sessions. Explicit suspension, delisting, and point-in-time universe policies remain future work.

Outputs:

- `predictions.csv`: out-of-sample scores, ranks, fold IDs, features, and realized labels where available.
- `folds.csv`: train/test boundaries and latest training-label date for leakage auditing.
- `equity.csv`: daily open-marked equity, cash, positions, and returns.
- `trades.csv`: signal/entry/exit dates, prices, weights, and net returns.
- `summary.json`: total return, open-marked maximum drawdown, daily Sharpe (252 sessions, zero risk-free rate), mean daily rank IC (Spearman), and trade count.

`config/settings.yaml` documents research defaults and `execution.enabled: false`; CLI flags are authoritative for the walk-forward runner. The YAML is not automatically loaded. The older `python -m models.train_lightgbm --csv ...` regression baseline remains available with corrected execution labels and boundary purging.

## Scope and verification

The tests cover bad data, download normalization/cache preservation with a mocked provider, feature causality, label alignment, strict fold purging, future-data invariance, deterministic ranking, portfolio costs/cash/non-overlap, and the paper-only broker. Synthetic test data verifies mechanics, not strategy profitability. No real historical performance claim is made.

A current hand-picked universe introduces survivorship bias. Retrospectively adjusted provider prices, fractional shares, fixed cost assumptions, absence of liquidity constraints, and open-only drawdowns limit realism. Compare against suitable benchmarks and add point-in-time membership, corporate-action/delisting handling, integer sizing, and calibrated costs before paper deployment.

## Source documentation

- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [LightGBM LGBMRanker](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)
- [NSE historical reports](https://www.nseindia.com/static/resources/historical-reports-capital-market-daily-monthly-archives)

## Seven-year Nifty 500 daily dataset

`data/nifty500/` contains an official constituent snapshot, NSE end-of-day data, an archive-date audit, and per-stock coverage. Daily records are stored in compressed yearly CSVs (`daily_YYYY.csv.gz`), which pandas reads directly: `pd.read_csv("data/nifty500/daily_2025.csv.gz")`. Only `date,symbol,open,close,volume` are retained. No minute or hourly data is collected.

```powershell
python -m data.nifty500_daily --start 2019-10-11 --end 2026-10-11 --output data/nifty500
```

End is exclusive. All calendar dates are checked, including possible special weekend sessions. The downloader resumes from git-ignored session caches. A request error prevents final dataset creation; HTTP 404s and archives containing another session date are recorded separately in `archive_status.csv`. These dates are not automatically declared holidays. Ordinary equity series (`EQ`, `BE`, `BZ`, `SM`, `ST`) and constituent REIT series (`RR`) are retained; changes to historical symbols or trading series are reflected as coverage gaps. Prices are as published by NSE (INR), without corporate-action adjustment; volume is total traded shares/units (`TTL_TRD_QNTY`).

The snapshot is the current Nifty 500 list, not its historical membership. New listings and symbol changes cause shorter coverage. `coverage.csv` records first/last dates, row counts, and missing observations relative to available archive sessions. Data is never backfilled or fabricated.

This open/close/volume dataset does not satisfy the existing OHLCV loader: high and low are required by ATR. Use it for open/close/volume analysis; corporate-action processing and a compatible feature set are needed before model training.

## Requested 2019–2022 swing features

[features/README.md](features/README.md) documents the ten indicator families and separate yearly files in `features/derived/`. Daily supporting OHLCV and the official Nifty 500 price-index closes are in `data/feature_inputs/`; December 2017 onward provides pre-2019 warm-up. Run `python -m features.generate_features` to regenerate offline. These files use unadjusted official prices and contain indicators only; the existing ranking/backtest feature selection is unchanged. Review corporate actions and historical universe membership before training on this dataset.
