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

## Persistent failed-download log and safe resume

The collector writes **two different kinds of audit data** to the external shared data folder:

- `reports/yahoo_finance/nifty50_download_errors.csv` — **append-only** persistent failure history, including the timestamp of the attempt, stock symbol, company name (when present in the constituent CSV), ISIN, Yahoo ticker, attempted year, requested `from_date` and `to_date`, and error description. Empty Yahoo responses are logged as `no_data_unverified`, not declared IPO gaps or successful downloads.
- `reports/yahoo_finance/nifty50_<SNAPSHOT>_<FROM>_<TO>_status.csv` — latest per-stock/year state (`downloaded`, `cached`, `no_data_unverified`, `error`). A diagnostic subset has a `_subset_...` suffix, so it does **not** overwrite the normal full-universe audit.

The collector already stores successful downloads in `candles/yahoo_finance/NSE_EQ/1d/<ISIN>/<YEAR>.csv.gz`. On the **next run with the same external data root**, it checks each existing partition's stock/ISIN/year identity and **does not download that partition again**. Only missing/failed/empty stock-years are attempted again. The old failure log is retained even after a retry succeeds. **Do not supply `--refresh`** unless you intentionally want to replace successfully cached data.

If every stock-year is failing, first try a tiny recent historical sample (completed calendar year):

```powershell
cd C:\path\to\ai-swing
git pull origin main
$env:INDIA_MARKET_DATA_HOME = 'D:\IndianMarketData'
python -m marketdata.yahoo_nifty50 `
  --universe-name nifty50 --from-date 2015-01-01 --to-date 2015-12-31 `
  --only-symbols RELIANCE TCS --pause-seconds 2
```

Inspect the persistent error file; it is populated as each attempt fails, even if the process is interrupted. Then run the full 2000–2016 command again without `--only-symbols` to resume. If you used `--asof` previously, provide the same value on subsequent runs and keep the constituent snapshot consistent. The targeted subset is for diagnostics; the training command **requires both complete full-universe audits** and cannot use a subset status file.

```powershell
Import-Csv 'D:\IndianMarketData\reports\yahoo_finance\nifty50_download_errors.csv' |
  Select-Object -Last 15 symbol,company_name,year,from_date,to_date,error_type,error_message |
  Format-Table -AutoSize
```

Common causes of all-stock errors include Yahoo rate limits, proxy/DNS/TLS failures, provider endpoint changes, or Python library version issues. The log shows which failure happened on your machine. A no-data year before a company's listing is not automatically considered an error, but it cannot be used to train the model. For live trading, Yahoo is not an exchange-approved execution feed.

## Download errors: human-managed `ignore=true` (Nifty 50 or Nifty 500)

The persistent, editable CSV for each basket is:

- `D:\IndianMarketData\reports\yahoo_finance\nifty50_download_errors.csv`
- `D:\IndianMarketData\reports\yahoo_finance\nifty500_download_errors.csv`

Open it in Excel, locate a stock-year **by both `isin` and `year`**, and set the new
`ignore` column to `true` or `false` (case insensitive). Close Excel before rerunning
Python. `ignore=true` skips that **Yahoo stock-year only**, even if the entire
period is requested again. `ignore=false` (the default) retries on future runs.
If multiple historical error rows exist for the same ISIN and year, edit the **last
matching row**, which determines the active policy. The existing CSV is upgraded
to add `ignore=false` without discarding prior errors or manual edits.

**Only set `ignore=true` when you have independently confirmed the period cannot
be downloaded (for example before the stock was listed).** Yahoo 429, 401/403,
timeouts, outages, and symbol-change issues should normally stay `false` so
other providers can recover them. Ignored years have status `ignored`, count
separately from `error` and `no_data_unverified` in the current report, and have
**no invented OHLCV**. When the Yahoo cache already contains valid data, the
existing successful partition is reused. Ignoring a Yahoo failure does **not**
block importing that stock-year from an alternative data vendor.

For your currently reported first-stage Nifty 50 collection: training-period
checks were 13 downloaded, 603 cached, 192 no-data, 42 errors; holdout checks
were 141 downloaded, 9 no-data, 0 errors. Confirm whether the 42 errors are
genuine non-listings or transient Yahoo failures before marking anything ignored.
After rerunning, `models.nifty50_stage1` will only accept a complete 850-check
2000–2016 report with zero current run errors. `ignored` years do not count as
successful data; they are explicit user-approved omissions.

## Download current Nifty 500 history from 2000, no broker account

From the `ai-swing` repository folder (PowerShell):

```powershell
# New basket has its own immutable constituent snapshot and audit reports;
# candles by ISIN/year are reused if already collected for Nifty 50.
git pull origin main
$env:INDIA_MARKET_DATA_HOME = 'D:\IndianMarketData'
$snapshot500 = Get-Date -Format 'yyyy-MM-dd'

python -m marketdata.yahoo_nifty50 --universe-name nifty500 `
  --from-date 2000-01-01 --to-date 2016-12-31 --asof $snapshot500

python -m marketdata.yahoo_nifty50 --universe-name nifty500 `
  --from-date 2017-01-01 --to-date 2019-12-31 --asof $snapshot500
```

The first downloader run retrieves a **current** Nifty 500 basket, not the
constituent list from 2000. Pre-listing years cannot have prices. Keep the same
`$snapshot500` for both runs, and do not use `--refresh` unless you deliberately
want to redownload historical partitions.

For a quick diagnostic subset without overwriting a full download audit:

```powershell
python -m marketdata.yahoo_nifty50 --universe-name nifty500 `
  --from-date 2015-01-01 --to-date 2015-12-31 --asof $snapshot500 `
  --only-symbols RELIANCE TCS
```

## Switch data providers without changing the model's input format

The normalized OHLCV schema **never depends on the source site**:
`date,symbol,isin,open,high,low,close,volume`.
Each vendor has a separate ISIN/year cache. No existing Yahoo partition is
replaced with NSE raw prices or a different vendor's prices. A research-ready
*source-selected copy* is kept independently under:

```
D:\IndianMarketData\candles\research_ohlcv\NSE_EQ\1d\<ISIN>\<YEAR>.csv.gz
D:\IndianMarketData\candles\research_ohlcv\NSE_EQ\1d\<ISIN>\<YEAR>.source.json
```

`source.json` records the originating vendor, original file checksum and the
selected normalized file checksum. Rerunning the build **does not change already
selected stock-year data**, even if you change provider priority. The AI model
always receives the same feature schema; changing the *data itself* can change
model outcomes and requires deliberate retraining/re-evaluation.

### First build the model-independent dataset from Yahoo history

```powershell
python -m marketdata.research_store --snapshot $snapshot500 `
  --universe-name nifty500 build `
  --from-date 2000-01-01 --to-date 2016-12-31 `
  --sources yahoo_finance

python -m marketdata.research_store --snapshot $snapshot500 `
  --universe-name nifty500 build `
  --from-date 2017-01-01 --to-date 2019-12-31 `
  --sources yahoo_finance
```

Inspect the resulting `reports\research_ohlcv\..._summary.json` and per-year
`..._status.csv` for `missing` and `errors`. Missing partitions remain missing,
not guessed. Build reports count every Nifty 500 stock-year checked.

### Later, fill missing Yahoo years from another provider

Obtain properly licensed, **corporate-action-adjusted daily OHLCV** as a CSV for
one stock and calendar year (e.g. `2013`); the file needs
`date,symbol,open,high,low,close,volume` (plus optional `isin`). For example:

```powershell
# This is an example of a user-supplied CSV, not a promised download service.
python -m marketdata.research_store --snapshot $snapshot500 `
  --universe-name nifty500 import `
  --source ext_alternative_vendor --csv 'D:\incoming\2013_RELIANCE.csv' `
  --confirm-adjusted-ohlcv

# Fill previously missing years from either vendor; already selected
# research_ohlcv partitions are reused rather than overwritten.
python -m marketdata.research_store --snapshot $snapshot500 `
  --universe-name nifty500 build `
  --from-date 2000-01-01 --to-date 2016-12-31 `
  --sources yahoo_finance ext_alternative_vendor
```

Unadjusted `nse_raw` is intentionally **not accepted** as adjusted training
history. The importer validates stock identity, price ranges, duplicate dates,
calendar years and records source checksums. It does not certify an external
provider's adjustment accuracy. Check splits, bonuses, symbol migrations,
missing sessions, data-use licences, and survivor bias before relying on returns.

### Train against the stable source-selected view

```powershell
python -m models.nifty50_stage1 --universe-name nifty500 `
  --snapshot $snapshot500 --price-source research_ohlcv `
  --allow-unverified-adjustments `
  --output artifacts/yahoo_nifty500_stage1
```

The same train/holdout split remains 2000–2016 vs. 2017–2019. New vendor data
is **not** automatically substituted into a previously trained model. The model
and outputs remain inside `ai-swing/artifacts`, while canonical candles reside
outside the repo for use by `MyTradingview` and other projects. Live orders stay
disabled.
## Kaggle historical CSV audit and recovery

See [Kaggle Nifty 50 import and safe recovery](KAGGLE_NIFTY50_IMPORT.md). Extract the Kaggle ZIP directly inside `INDIA_MARKET_DATA_HOME/kaggle`; run `python -m marketdata.kaggle_nifty50 --snapshot <the original snapshot date>` to compare Kaggle raw OHLCV with existing Yahoo adjusted history. Kaggle is **not an adjusted-price replacement**. The importer writes its own `kaggle_raw` cache and reports; it does not change Yahoo, `research_ohlcv`, or trained models.
