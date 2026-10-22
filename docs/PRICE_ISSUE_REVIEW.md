# ai-swing: price-issue review and repair

This is a CSV round-trip workflow. All files remain on your local Windows machine
until you choose to upload the review CSV to ChatGPT.

## Installation
Save `price_issue_review.py` into the repository's `marketdata/` directory.

## 1. Generate a populated issue CSV

```powershell
cd "D:\Work\Project\ai-swing\ai-swing"
$env:INDIA_MARKET_DATA_HOME = "D:\Work\Project\IndianMarketData"
python -m marketdata.price_issue_review --snapshot 2026-10-18 --universe-name nifty50 scan --provider research_v3
```

The output appears at:
`D:\Work\Project\IndianMarketData\reports\price_issue_review\research_v3_nifty50_2026-10-18_issues.csv`

Only run `scan` once for each review; it refuses to overwrite an existing edited CSV.

## 2. Upload that CSV to ChatGPT

Ask ChatGPT to verify suspicious dates and prices against relevant historical
NSE/exchange reports and corporate-action notices. Evidence may be missing:
an unresolved issue MUST remain `verification_result=pending` or `unresolved`,
with `correction_action=none`. ChatGPT should preserve **every row**, the issue
ID, original prices, and SHA-256 columns. A returned CSV must use identical
columns and may add only reviewer findings in the existing editable columns.

| Review field | Allowed values / usage |
| --- | --- |
| `verification_result` | `pending`, `verified_issue`, `false_positive`, `unresolved` |
| `verification_notes` | Rationale and quality of independent cross-check |
| `reference_url` | Exact evidence URL; mandatory for actual corrections |
| `reviewed_by` | Optional reviewer identifier |
| `correction_action` | `none`, `keep_original`, `replace_bar`, `exclude_stock_year` |
| `corrected_open/high/low/close/volume` | Supply ALL FIVE numeric values only for `replace_bar` |

Only `verified_issue` with a reference URL and review notes can result in `replace_bar`
or `exclude_stock_year`. `keep_original` requires `false_positive`.

**CAUTION:** A stock split/bonus or Yahoo–Kaggle basis mismatch can affect many
dates and entire historical regimes. Never guess a one-day price to remove an
artificial jump. Use `exclude_stock_year` pending a verified, consistently adjusted
history, or leave the issue `unresolved`.

## 3. Replace CSV and dry-run review

Download the reviewed CSV from ChatGPT and replace the exact same local file.
Close Excel before running.

```powershell
$csv = "$env:INDIA_MARKET_DATA_HOME\reports\price_issue_review\research_v3_nifty50_2026-10-18_issues.csv"

python -m marketdata.price_issue_review --snapshot 2026-10-18 --universe-name nifty50 apply `
  --parent research_v3 --version research_v4 --review-file $csv --dry-run
```

## 4. Apply verified repairs to NEW research_v4

```powershell
python -m marketdata.price_issue_review --snapshot 2026-10-18 --universe-name nifty50 apply `
  --parent research_v3 --version research_v4 --review-file $csv
```

This NEVER overwrites raw Kaggle, Yahoo, or research_v3. The program refuses
to replace research_v4 once created. To make another change, choose a new
version (e.g. research_v5) and scan the new parent version.

Output is in:
`$env:INDIA_MARKET_DATA_HOME\candles\research_v4\...`
and
`$env:INDIA_MARKET_DATA_HOME\reports\research_v4\price_issue_review_result.json`

Train a new model only after confirming the remaining abnormal transitions.
A clean checksum or a verified single-bar repair does NOT independently
certify corporate-action adjustment consistency.