# Official daily supporting inputs

`ohlcv_2017.csv.gz` through `ohlcv_2022.csv.gz` contain date, symbol, daily open/high/low/close (INR), and total traded shares/units. The 2017 file covers December only. `nifty500_index.csv` contains official daily Nifty 500 price-index closes on available sessions. No intraday data is used.

The stock source is NSE's historical daily capital-market bhavcopy ZIP; the benchmark source is NSE's all-index daily closing report. `manifest.json` records exact URL templates, dates, retrieval time, annual counts, and hashes. `download_status.csv` audits every calendar date, including weekends. A missing archive is reported as unavailable rather than inferred to be a holiday. HTTP errors fail final generation and can be retried from the git-ignored cache.

```sh
python -m data.feature_inputs --start 2017-12-01 --end 2023-01-01 --workers 16
```

End is exclusive. The current Nifty 500 constituent snapshot in `data/nifty500/constituents.csv` defines the universe; absent, renamed, and newly listed securities are not fabricated. Series EQ/BE/BZ/SM/ST/RR are included. Supporting open/close/volume fields must exactly reconcile with the original dataset over shared dates before yearly files are finalized.

These are as-published, unadjusted prices. Corporate actions can distort indicators; point-in-time constituent membership and corporate-action reconciliation are separate research requirements. Zero-price inactive records are retained for provenance but are excluded from indicator history. The original `data/nifty500/` files are unchanged.
