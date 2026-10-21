# Kaggle Nifty 50 comparison and safe recovery (no broker account)

The source archive is Rohan Rao's `NIFTY-50 Stock Market Data (2000–2021)`. It contains individual stock files (`RELIANCE.csv`, `TATAMOTORS.csv`, `MM.csv`, ...), `stock_metadata.csv` and a duplicated combined file `NIFTY50_all.csv`. **Do not import the combined file**, because the individual files supply the same records and the metadata identifies stocks by ISIN. `INFRATEL.csv` is empty in the supplied archive.

## 1. Extract the ZIP outside the repository

In PowerShell, from the `ai-swing` root:

```powershell
$env:INDIA_MARKET_DATA_HOME = 'D:\IndianMarketData'
New-Item -ItemType Directory -Force 'D:\IndianMarketData\kaggle' | Out-Null
Expand-Archive -LiteralPath 'D:\Downloads\archive.zip' -DestinationPath 'D:\IndianMarketData\kaggle' -Force
```

If your ZIP is at another location, replace `D:\Downloads\archive.zip`. Files should be directly under `D:\IndianMarketData\kaggle\stock_metadata.csv` and `D:\IndianMarketData\kaggle\RELIANCE.csv`, **not an extra nested folder**.

Find your existing frozen Nifty 50 snapshot (use the same date as earlier training):

```powershell
$snapshot = '2026-10-18'  # replace if your original snapshot was another date
python -m marketdata.kaggle_nifty50 --snapshot $snapshot --universe-name nifty50
```

The importer is offline. It checks the snapshot's stock symbols and ISINs against `stock_metadata.csv` and allows historical ticker names inside files (e.g. `TELCO` -> `TATAMOTORS`, `UTIBANK` -> `AXISBANK`). The file for `M&M` is named `MM.csv`. Companies not included in the frozen Nifty 50 snapshot are reported as unmatched, **not** silently added to your training universe.

## 2. Review the audit

The command saves validated Kaggle **unadjusted** records separately, without modifying Yahoo, under:

```
D:\IndianMarketData\kaggle\                 # original extracted CSVs
D:\IndianMarketData\candles\kaggle_raw\NSE_EQ\1d\<ISIN>\<year>.csv.gz
D:\IndianMarketData\reports\kaggle_raw\nifty50_<snapshot>_summary.json
D:\IndianMarketData\reports\kaggle_raw\nifty50_<snapshot>_coverage.csv
D:\IndianMarketData\reports\kaggle_raw\nifty50_<snapshot>_comparisons.csv
D:\IndianMarketData\reports\kaggle_raw\nifty50_<snapshot>_missing_yahoo_dates.csv
D:\IndianMarketData\reports\kaggle_raw\nifty50_<snapshot>_errors.csv
```

`comparisons.csv` gives yearly counts of dates present only in Kaggle versus Yahoo, matching-date counts and adjusted-vs-unadjusted price differences. `missing_yahoo_dates.csv` lists exact dates that *might* be candidates for repair. A raw price discrepancy is **not automatically an error**: splits, bonuses, dividends, different adjustment conventions, and historic corporate identities cause genuine differences.

Kaggle source files are retained in `kaggle/`. The importer validates that individual files match metadata ISINs; invalid OHLCV and identity mismatches are reported. Rerunning the command reuses matching saved `kaggle_raw` partitions rather than overwriting them. `--no-cache-raw` runs only comparisons, without writing new Kaggle cache partitions.

## 3. Updating model data safely

**Kaggle historical Open/High/Low/Close prices are raw (not guaranteed corporate-action-adjusted). Do not directly overwrite `candles/yahoo_finance/...` or `candles/research_ohlcv/...` with them.** A wrong split adjustment can create artificial 50%+ returns and mislead the model. The audit intentionally does not modify either of those directories or your trained model.

When `..._missing_yahoo_dates.csv` identifies useful historical records, first verify the instrument identity, price adjustment, split/bonus dates and Yahoo overlaps. If you have prepared **independently verified adjusted** daily OHLCV for one stock in one year, save a CSV with `date,symbol,open,high,low,close,volume`. Then import via the existing provider-neutral interface:

```powershell
python -m marketdata.research_store --snapshot $snapshot --universe-name nifty50 import `
  --source ext_kaggle_adjusted --csv 'D:\VerifiedOHLCV\RELIANCE_2013_adjusted.csv' `
  --confirm-adjusted-ohlcv
```

The confirmation flag is an explicit declaration from you and **does not itself adjust or validate corporate actions**. After importing approved adjusted stock-years, rebuild the *missing* research partitions, without `--refresh`:

```powershell
python -m marketdata.research_store --snapshot $snapshot --universe-name nifty50 build `
  --from-date 2000-01-01 --to-date 2016-12-31 `
  --sources yahoo_finance ext_kaggle_adjusted

python -m marketdata.research_store --snapshot $snapshot --universe-name nifty50 build `
  --from-date 2017-01-01 --to-date 2019-12-31 `
  --sources yahoo_finance ext_kaggle_adjusted
```

**Note:** The research-store builder works at the **stock/year partition level**, not day-by-day. If a Yahoo partition already exists, it will remain unchanged, even if Kaggle has dates missing within that Yahoo year. Improving within-year gaps requires deliberate creation and verification of a complete adjusted replacement partition, then an *explicit separate versioned rebuild*. Never silently splice raw Kaggle prices into an adjusted Yahoo series.

Train or evaluate a **new** model after any approved adjustments, under a new output directory:

```powershell
python -m models.nifty50_stage1 --snapshot $snapshot --universe-name nifty50 `
  --price-source research_ohlcv --allow-unverified-adjustments `
  --output artifacts/nifty50_kaggle_reviewed
```

The original model remains unchanged until you choose to retrain; a provider change can change model results even if the column names stay the same. Reserve 2020+ for truly new validation and keep live trading disabled.
