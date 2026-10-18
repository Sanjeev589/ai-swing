# No-broker-account NSE daily data: Yahoo + NSE (Nifty 50 first)

**Entry point:** `marketdata/yahoo_nifty50.py` (run as `python -m marketdata.yahoo_nifty50`).  
**No Upstox account, Upstox token, or live broker access needed.** Research only; no live trading.

## Windows setup, Nifty 50 2000–2019

Run these commands **from the ai-swing repository root** using Python 3.11 or 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:INDIA_MARKET_DATA_HOME = 'D:\IndianMarketData'  # must be outside repo
$snapshot = Get-Date -Format 'yyyy-MM-dd'

# Stage 1: historical training inputs (inclusive calendar-year endpoints)
.\.venv\Scripts\python.exe -m marketdata.yahoo_nifty50 `
  --universe-name nifty50 --from-date 2000-01-01 --to-date 2016-12-31 --asof $snapshot

# Stage 2: untouched initial evaluation inputs (same universe snapshot)
.\.venv\Scripts\python.exe -m marketdata.yahoo_nifty50 `
  --universe-name nifty50 --from-date 2017-01-01 --to-date 2019-12-31 --asof $snapshot

# Stage 3: strictly evaluate the split; explicit unverified-adjustment acknowledgement
.\.venv\Scripts\python.exe -m models.nifty50_stage1 `
  --universe-name nifty50 --snapshot $snapshot --allow-unverified-adjustments
```

The downloaded data is usable even if you do not run model training. The `--asof` flag is a **stock-membership snapshot label**; it is not the historical-data end date. Do not replace the same dated snapshot with a different constituent list. If NSE blocks the public CSV, download its constituent CSV manually and add `--universe-csv path\to\nifty50list.csv` to both Yahoo commands. For Yahoo renamed/ticker differences use `--overrides-csv path\to\overrides.csv` with `symbol,yahoo_ticker` columns, and use the *same mapping* for both phases. If data is missing, check the audit reports; do not fill missing history with artificial prices.

### Where the reusable files go

```
D:\IndianMarketData\
  universes\nifty50\YYYY-MM-DD.csv                 # frozen membership snapshot: symbol, ISIN, Yahoo ticker
  universes\nifty50\YYYY-MM-DD.json                # source and checksum
  candles\yahoo_finance\NSE_EQ\1d\<ISIN>\<YEAR>.csv.gz
  candles\nse_raw\NSE_EQ\1d\<ISIN>\<YEAR>.csv.gz
  reports\yahoo_finance\nifty50_..._status.csv     # stock-by-year success/empty/error audit
  reports\yahoo_finance\nifty50_..._summary.json   # request counts and provenance
  reports\nse_raw\nifty50_..._comparisons.csv   # raw vs adjusted OHLC differences
```

Every saved candle row uses `date,symbol,isin,open,high,low,close,volume` with daily NSE trading dates. The data store is intentionally outside GitHub. It can be read by `MyTradingview`, `ai-swing` and any other Python project via `pd.read_csv('...csv.gz', parse_dates=['date'])`. Source/adjustment provenance remains distinct. Yahoo `auto_adjust=True` does not make a promise of fully verified historical corporate actions or volume adjustments. NSE bhavcopy prices are as published (unadjusted); **never overwrite the Yahoo historical series with NSE raw prices.** Treat percentage discrepancies as a review flag, not necessarily an error.

### Checking one NSE trading session

To import an official NSE daily ZIP/CSV downloaded manually (most reliable when historical archive endpoints change):

```powershell
.\.venv\Scripts\python.exe -m marketdata.nse_daily `
  --universe-name nifty50 --snapshot $snapshot --date 2019-12-31 `
  --archive 'D:\Downloads\BhavCopy_NSE_CM_0_0_0_20191231_F_0000.csv.zip'
```

Omit `--archive` to attempt the currently configured public NSE archive URL. Missing archive files, HTTP errors, schema changes and mismatched ISINs are reported as errors, not silently substituted. **NSE raw daily data is independent corroboration and daily update input, not a replacement for fully adjusted training prices.** This downloader is for completed trading sessions; it is not a real-time broker feed.

## Add other stock universes later, without modifying source code

Default for now: **Nifty 50 only**. Later, two options are supported:

- Automatically fetch a **current Nifty 500** snapshot using `--universe-name nifty500`.
- Use any user-supplied basket CSV with `Symbol,ISIN Code` and a safe name, e.g. `--universe-name midcaps_2026 --universe-csv D:\lists\midcaps.csv`. The `--universe-name` value must use lowercase letters/digits/underscores/hyphens and is used to separate membership snapshots and audit reports.

Example of the *later* custom workflow:

```powershell
.\.venv\Scripts\python.exe -m marketdata.yahoo_nifty50 `
  --universe-name midcaps_2026 --universe-csv D:\lists\midcaps.csv `
  --from-date 2000-01-01 --to-date 2016-12-31 --asof $snapshot
```

The same ISIN/year candles can be shared across baskets. Different basket definitions must use different names or dated snapshots. Future validation can use `python -m models.nifty50_stage1 --universe-name midcaps_2026 ...` after collecting *both* historical ranges for that basket. Do not use a later basket to claim period-correct index membership: it is a current-stock snapshot research experiment.

## What training produces and what it cannot prove

`models/nifty50_stage1.py` consumes **only** the external Yahoo `yahoo_finance` OHLCV partitions, verifies complete per-year request audits, trains LightGBM LambdaRank only on labels matured by 2016-12-31 and scores the untouched 2017–2019 holdout. It compares against a 20-day momentum rank with the same five-session non-overlapping hypothetical cohorts and illustrative 25 bps costs per side. It writes its **model and predictions inside ignored `artifacts/yahoo_nifty50_stage1/`**, not in the shared market data folder. The model remains research-only and does not submit orders.

Historical limitations: listed-after-2000 stocks lack earlier observations; present-day Nifty membership creates survivorship bias; ticker/ISIN changes may require explicit correction; Yahoo downloads may fail or throttle; corporate actions must be checked independently; trade simulations are simplified and do not account for overnight liquidity shocks, actual taxes, order-book impact or intraday stops. No claim of profitability or verified live win rate is made. **Do not repeatedly tune on the 2017–2019 holdout.** Keep 2020+ unopened until choosing a model using earlier data.

Official primary sources: [Nifty 50 constituents](https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv), [Nifty 500 constituents](https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv), [NSE market reports](https://www.nseindia.com/all-reports), and [yfinance project](https://github.com/ranaroussi/yfinance). Respect all provider and exchange data-usage terms and redistribution rules.
