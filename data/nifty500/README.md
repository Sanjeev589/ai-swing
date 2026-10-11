# Nifty 500 daily open, close and volume

Official NSE daily bhavcopy records for the current Nifty 500 constituent snapshot. The requested window is 11 October 2019 through 10 October 2026; see `manifest.json` for the actual first/last available sessions and validated totals.

Each `daily_YYYY.csv.gz` contains only `date,symbol,open,close,volume`. Prices are published NSE prices in INR, without corporate-action adjustment. Volume is daily traded shares/units. There is no minute or hourly data.

```python
from pathlib import Path
import pandas as pd

prices = pd.concat(
    [pd.read_csv(path, parse_dates=["date"])
     for path in sorted(Path("data/nifty500").glob("daily_*.csv.gz"))],
    ignore_index=True,
)
```

- `constituents.csv`: official current constituent list, including source series and ISINs.
- `coverage.csv`: each constituent's first/last date, row count, and missing observations relative to available sessions.
- `archive_status.csv`: audit of every calendar date requested, including missing archives and wrong-date responses.
- `manifest.json`: provenance, date boundaries, row counts, price conventions and file SHA-256 checksums.

The official snapshot includes a non-traded temporary Dummy HEG entry. No prices are fabricated for it. Newly listed securities have shorter histories. Historical index membership and corporate-action adjustments are not supplied.

The existing feature pipeline requires high/low for ATR, so these open/close/volume files are a separate raw research dataset.
