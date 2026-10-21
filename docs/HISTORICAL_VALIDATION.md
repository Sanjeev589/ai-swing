# Historical security validation and versioned training-data recovery

This is an **offline review and controlled-data publication stage** for `ai-swing`.
Do not overwrite the original Yahoo-adjusted candles. You can **explicitly select**
`kaggle_raw` for a stock-year in the review CSV for exploratory research. Such
prices are UNADJUSTED, and mixing them with Yahoo-adjusted history can create
artificial returns and technical indicators. The new versioned model dataset
records source and adjustment warnings; it is not automatically corrected.

It preserves three separate inputs under `INDIA_MARKET_DATA_HOME`:

- `candles/yahoo_finance/...`: original Yahoo-adjusted cache, unchanged.
- `candles/kaggle_raw/...`: original imported Kaggle raw OHLCV, unchanged.
- `candles/ext_<vendor>/...`: independently reviewed adjusted vendor CSVs, if imported.

Approved stock-years go into `candles/research_v2/NSE_EQ/1d/<ISIN>/<year>.csv.gz`;
`*.source.json` next to each year records the price source, SHA-256 checksums,
listing gate, any pre-listing rows trimmed, review note, and cited evidence.
An existing version is never silently overwritten; use `v3` for a new revision.

## 1. Review your existing 2000–2019 Nifty 50 data

In Windows PowerShell at the `ai-swing` repository root:

```powershell
 git pull origin main
 $env:INDIA_MARKET_DATA_HOME = 'D:\Work\Project\IndianMarketData'
 python -m marketdata.historical_validation `
   --snapshot 2026-10-18 --universe-name nifty50 --version v2 audit
```

Inspect the generated files in
`D:\Work\Project\IndianMarketData\reports\historical_validation`:

- `nifty50_2026-10-18_v2_review.csv`: **editable** stock/year decisions.
- `listing_rules.csv`: symbol, earliest verified listed date, and official evidence.
- `corporate_actions.csv`: dated documented corporate actions; add validated ones.
- `nifty50_2026-10-18_v2_review_corporate_actions.csv`: price-ratio observations
  immediately before/after listed corporate-action dates where comparable.
- `nifty50_2026-10-18_v2_review.json`: review counts and paths.

The first rules contain **TCS (2004-08-25)** and **BAJAJ-AUTO / BAJAJFINSV
(2008-05-26)** supported by NSE exchange announcements. TCS bonus dates
2006-08-09, 2009-06-18, and 2018-06-04 are sourced from TCS investor relations.
These *do not* imply other stock listing histories or other corporate actions
have been verified. Add more sourced rules carefully. An ISIN mismatch is a
**manual historical-security identity investigation**, not an automatic match.

### 2. Edit the review CSV (close Excel before running the script)

Every stock/calendar-year row has `decision`:

| decision | What it means |
| --- | --- |
| `requires_review` | Default: this year CANNOT enter the reviewed model data. |
| `must_exclude` | Exclude the year, e.g. confirmed pre-listing/no trustworthy adjusted data. |
| `approved_adjusted` | Explicitly choose a provider (adjusted Yahoo, reviewed external, or deliberately raw Kaggle) for a stock-year. With Kaggle, this label does **not** mean prices were adjusted. |

For `approved_adjusted` fill in:

- `approved_source`: `yahoo_finance`, `kaggle_raw`, or an existing reviewed import `ext_<vendor>`.
  `nse_raw` and `research_ohlcv` remain blocked. Only select `kaggle_raw` if you
  intentionally accept raw historical prices and their corporate-action risks.
  The matching Kaggle ISIN must be present in `kaggle/stock_metadata.csv`, and
  `candles/kaggle_raw` must have been populated by the Kaggle audit.
- `evidence_url` **(optional)**: supporting information (exchange, corporate action, vendor methodology).
- `review_note` **(optional)**: your reason and adjustment verification. The program cannot
  independently certify price adjustments merely from these fields. Both may be left
  blank to allow publishing, but the resulting summary reports the number of
  undocumented approvals and each partition records `approval_documented=false`.
  Blank fields **do not** establish that price adjustments were verified.
- `trim_prelisting`: retained for compatibility, but no longer required.
  For an approved stock-year with a known verified listing date, the build
  automatically removes pre-listing rows from the new research copy. The source
  CSV/GZIP is never modified. The build report includes the exact number
  of automatically trimmed rows and examples.

The `finding`, `isin_mismatch`, `kaggle_only_dates`, `max_abs_ratio_step_pct`,
`yahoo_large_jumps`, and prelisting counts help prioritize your review. A large
Kaggle/Yahoo price difference is normal when their corporate-action adjustment
bases differ. Do not approve on overlap closeness alone.

The audit command preserves previous manual decisions only when the relevant
Yahoo checksum, Kaggle identity, and verified listing date have not changed.

### 3. Optional: Import independently adjusted missing history

If you prefer adjusted rather than explicitly raw Kaggle history, obtain/prepare
reviewed corporate-action-adjusted OHLCV for one stock/year and import it using
the existing importer:

```powershell
python -m marketdata.research_store `
  --snapshot 2026-10-18 --universe-name nifty50 import `
  --source ext_reviewed_vendor --csv 'D:\Incoming\2005_ITC_adjusted.csv' `
  --confirm-adjusted-ohlcv
```

The source CSV needs `date,symbol,open,high,low,close,volume` and may include
`isin`. The confirmation flag records your assertion, not an independent audit.
Then set that stock-year's `approved_source=ext_reviewed_vendor` and provide
supporting evidence in the review file.

### 4. Publish reviewed v2 dataset (after all rows have decisions)

```powershell
python -m marketdata.historical_validation `
  --snapshot 2026-10-18 --universe-name nifty50 --version v2 build `
  --from-date 2000-01-01 --to-date 2016-12-31

python -m marketdata.historical_validation `
  --snapshot 2026-10-18 --universe-name nifty50 --version v2 build `
  --from-date 2017-01-01 --to-date 2019-12-31
```

Publishing **fails closed** if a reviewed period contains even one
`requires_review` row, if selected source data is absent or mismatched by ISIN,
while requiring explicit review of remaining stock-year sources.
Fully pre-listing stock-years are **automatically excluded by verified listing
rules**, even if an earlier manual decision approved them; the build summary
reports `automatically_excluded_prelisting_stock_years` and sample identities.
The review CSV is not modified by the build. Listing-year records containing
only pre-listing sessions are also excluded. For approved listing-year records with
both pre- and post-listing sessions, the build automatically trims the invalid
earlier rows without requiring `trim_prelisting=true`. It reports
`automatically_trimmed_prelisting_stock_years`, `automatically_trimmed_prelisting_rows`
and sample symbols, and stores a trim count in each research partition's metadata.
Only independently verified dates in `listing_rules.csv` trigger this behavior.
Evidence URL and note are optional. Selecting `kaggle_raw` writes warnings and unadjusted provenance,
but does not block the build. If the first run finds pending rows, fill the review CSV first; no
partitions from that run are published. Reviewed omissions remain absent,
**not synthesized**. Revisions after publication require a new version (`v3`).

### 5. Train and validate a SEPARATE Nifty 50 model

```powershell
python -m models.nifty50_stage1 `
  --universe-name nifty50 --snapshot 2026-10-18 `
  --price-source research_v2 --allow-unverified-adjustments `
  --output artifacts/nifty50_reviewed_v2
```

Both versioned audit summaries must have complete 850/150 stock-year checks,
zero errors, zero pending rows, and **the same reviewed manifest hash**.
The model also checks each canonical candle's recorded SHA-256 and source approval,
including explicit Kaggle raw provenance.
It writes `dataset_fingerprint_sha256` to the model report. The old trained
model and baseline reports remain untouched.

**Important interpretation:** `approved_adjusted` means *selected by the operator*,
not guaranteed exchange-verified or adjusted. When Kaggle raw is included,
`raw_kaggle_stock_years` and an explicit warning appear in the research build
and model reports. Never interpret a raw/Yahoo-adjusted price discontinuity as
a real market return. The current trading simulation still has
survivorship bias and simplified next-open fills; it does not enable live trading.
The 2017–2019 holdout should not be used repeatedly to select hyperparameters.
