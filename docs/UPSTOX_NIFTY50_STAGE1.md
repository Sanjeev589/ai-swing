# Upstox V3 — current Nifty 50, reusable daily data, staged AI research

**This is research software. No live execution, no order API calls, and no promise of profitable returns.**

## Shared dataset outside the project

Choose a neutral location usable by `ai-swing`, `MyTradingview`, and other local projects, e.g. Windows `D:\IndianMarketData`. This path is **not** a folder inside `ai-swing` and should not be committed to GitHub.

```powershell
$env:INDIA_MARKET_DATA_HOME = 'D:\IndianMarketData'
$env:UPSTOX_ACCESS_TOKEN = '<your local Upstox access token>'
```

Get an Upstox app and authentication token through the official developer dashboard. Do not send tokens in ChatGPT, put them in scripts, or commit them. Reauthenticate as required by Upstox. Reuse or redistribute broker market data only as allowed by the broker's and exchange's terms.

The shared data library is independent of AI model code:

```
D:\IndianMarketData\
  universes\nifty50\2026-10-18.csv           # frozen current-members snapshot
  universes\nifty50\2026-10-18.json          # source links / checksum
  candles\upstox_v3\NSE_EQ\1d\INE...\2000.csv.gz  # one ISIN/year, full OHLCV
  reports\upstox_v3\nifty50_..._status.csv  # cached/downloaded/empty/error
  reports\upstox_v3\nifty50_..._summary.json
```

Each candle file has `date,symbol,isin,instrument_key,open,high,low,close,volume,open_interest`. Session dates are NSE India local dates. Dates, prices, duplicate rows and instrument identities are validated. Repeated runs reuse full-year partitions, with atomic writes. The universe CSV records the **current** 50 stocks and must not be misrepresented as historical Nifty50 membership.

## Phase 1: acquire 2000–2016 daily OHLCV, then 2017–2019 validation data

```powershell
python -m marketdata.upstox_v3 --from-date 2000-01-01 --to-date 2016-12-31 --asof 2026-10-18
python -m marketdata.upstox_v3 --from-date 2017-01-01 --to-date 2019-12-31 --asof 2026-10-18
```

The `--asof` date should correspond to the **actual date you fetch the constituents and Upstox instrument master**. If using a different day, omit `--asof` to create a correctly dated snapshot. Reusing that same dated snapshot is possible with the offline `--universe-csv` and `--instrument-json` options. Never silently rewrite a previous snapshot.

Upstox V3 historical daily API advertises availability *from January 2000* and a *maximum ten-year window per request*. This integration requests by calendar year, keeping within the limit; **no guarantee** is made that every current Nifty 50 stock has data starting in 2000. Older stocks may have corporate actions, ticker/ISIN transitions and shorter histories. The shared report counts successful, cached, empty and failed requests **separately**. HTTP 401/403 aborts the sweep rather than producing fictional bars.

## Phase 1: freeze training through 2016 and evaluate 2017–2019

```powershell
python -m models.upstox_nifty50_stage1 --snapshot 2026-10-18 --allow-unverified-corporate-actions --output artifacts/upstox_nifty50_stage1
```

The adjustment flag is an explicit acknowledgement that **Upstox V3 OHLCV corporate-action normalization is not independently verified**; output is therefore exploratory and not ready for live use. For a real-money strategy, reconcile adjusted OHLCV, split/bonus events, historical membership, old ISINs and exchange symbols first. Retain 2020 onward for a genuinely unseen subsequent test.

The training uses existing causal daily features and LightGBM LambdaRank. Labels are next-session open to open at five sessions later (by default). Training labels are purged to finish no later than 2016-12-31; holdout outcomes run in 2017–2019 only. Comparisons use the **same** candidate universe, the same nonoverlapping five-session cohorts and an estimated 25 basis points cost per side. Metrics include completed trades, wins, losses, win rate, cost-adjusted compounded cohort returns, and cohort drawdown, along with direct relative-momentum baseline. Losses are losing trades, **not program errors**. This initial holdout must not be repeatedly optimized; tune using earlier training-only validation and reserve 2020+ for a fresh blind test.

Artifacts remain in `ai-swing/artifacts/`, while shared raw/normalized candles stay in `D:\IndianMarketData`.

### Known shortcomings

- Current constituent selection causes survivor/selection bias when retrospectively tested in 2000–2019; these results do not simulate the actual historical Nifty 50 portfolio.
- Upstox may not expose old/superseded identities through today's instrument master; **never synthesize their missing history**.
- Verify adjusted price and volume before trusting technical indicators and trade P&L.
- Model experiment uses fixed-horizon simulated entries and exits, not intraday ATR stops. Additional market-regime/risk simulation can be tested after integrity audits.
- No live trades or broker credentials saved. No post-2019 data is collected automatically until the first-stage results are reviewed.

Official references: [Historical Candle V3](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/), [Instruments](https://upstox.com/developer/api-documentation/instruments/), [Nifty 50 constituents](https://www.niftyindices.com/indices/equity/broad-based-indices/nifty--50), [Upstox rate limits](https://upstox.com/developer/api-documentation/rate-limiting/).