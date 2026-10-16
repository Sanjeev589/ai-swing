# Market-regime + LightGBM Nifty 500 swing research (paper only)

This **separate research runner** compares the existing LightGBM LambdaRank architecture with a **non-AI momentum rule** on identical eligible breakout/pullback setups. It does not replace the legacy 2019–2022 trained model or execute any real orders.

## Inputs and safety

- Daily **open, high, low, close, volume** per symbol, including a warm-up period of at least 252 trading days. Paths may be a CSV or a directory containing `ohlcv_YYYY.csv.gz` files. An open/close-only dataset **will be rejected**, not silently approximated.
- Index CSV: `date,nifty500_close`; this is a **price index**, not the Nifty 500 TRI.
- The checked-in `data/feature_inputs/` OHLCV dataset stops at **2022** and contains **unadjusted NSE prices** and a current constituent snapshot. It is usable only in explicitly flagged exploratory mode, not for reliable post-2022 claims.
- `--price-policy verified-adjusted` is a **user attestation** that inputs are corporate-action-adjusted, not a verifier. If a nearby source manifest declares unadjusted prices, the runner rejects that attestation. `--price-policy research-unadjusted` accepts known-unadjusted inputs with a strong report warning.
- Historical point-in-time constituents, splits/bonus/dividend adjustments, suspension/delisting handling and data rights should be independently resolved before interpreting model results. Data gaps inside open positions **raise an error**.
- No broker connection exists. `execution.enabled: false` remains unchanged.

## Example: saved NSE historical study (2019–2021 training, 2022 testing)

```powershell
python -m backtesting.ai_swing `
  --prices data/feature_inputs `
  --benchmark data/feature_inputs/nifty500_index.csv `
  --price-policy research-unadjusted `
  --test-start 2022-01-03 --test-end 2022-12-30 `
  --output artifacts/ai_swing_2022
```

For 2023–2026, **provide validated OHLCV** for those years plus the corresponding Nifty 500 index calendar. The separate `data/nifty500/daily_YYYY.csv.gz` files cannot substitute for those OHLCV inputs because they lack `high` and `low`.

Optional sectors: `--sectors data/local/sectors.csv`, with `symbol,sector` columns. Static sector mapping is not automatically point-in-time.

Use `--help` to inspect thresholds. Example overrides: `--horizon 5 --max-hold-sessions 10 --cost-bps-per-side 35 --risk-per-trade 0.004`.

## Signal and portfolio rules

1. Market regime: Nifty 500 above its 200-day average, positive 20-session index momentum and breadth above a configurable threshold (minimum stock count required).
2. Liquidity/quality: 20-session traded value, 14-day ATR fraction, no major 20-session price discontinuity, and complete indicator warm-up.
3. Setup: either an EMA21/EMA50 uptrend with a prior 20-session high breakout plus volume/ADX/relative strength, or a pullback toward EMA21 with recovery, RSI and relative strength confirmation.
4. LightGBM model trains on next-open-to-future-open return ranks, **never on labels that mature on/after the next fold's first test signal**. Refitting every 63 sessions is supported, with deterministic equal-score ordering by symbol.
5. Comparator: the **same candidates and dates**, ranked solely by relative strength and the breakout indicator. Only the rank policy differs; this enables an AI-ablation comparison.
6. Trade: signal at session close, next-day opening entry; integer shares, capped positions/exposure, ATR-based initial stops, explicit gap losses, next-day-only trailing-stop updates, maximum holding period, charges on both sides, and cash for unused capacity. Stops inferred from daily OHLC are approximations; intraday fills may be worse.

## Outputs

`summary.json`, `folds.csv`, `out_of_sample_signals.csv`, plus `ai_score_equity.csv`, `ai_score_trades.csv`, `rule_score_equity.csv`, and `rule_score_trades.csv`.

Metrics include completed trades, winning/losing trades, win rate, net return in INR and percent, Sharpe, max drawdown, rejected gaps and profit factor. No claim of a 55–65% or higher hit rate is made. **Select an AI strategy only if it beats the rule baseline on realistic unseen, after-cost performance, not merely on in-sample fit.**

### Model limitations

The ranking target is open-to-open forward return, whereas portfolio exits include ATR stops and holding-period limits; they are not identical objectives. Sector strength is optional because no reliable point-in-time sector map is included. Fee/slippage is a single configurable bps approximation, not a complete Indian broker/tax schedule. Momentum 50 TRI and real historical membership must be added before final production-comparison claims.