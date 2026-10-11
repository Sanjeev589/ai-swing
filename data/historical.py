"""Optional Yahoo Finance research adapter for NSE (.NS), with explicit CSV cache."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data.nse_loader import validate_daily


def download_nse(symbols, start, end, output, *, downloader=None):
    """Fetch adjusted daily prices, start inclusive/end exclusive. Fail on any missing symbol.

    Yahoo is a third-party research source, not an official NSE licensed feed.
    The cache is only written after every requested symbol validates successfully.
    """
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must precede end")
    symbols = list(dict.fromkeys(str(s).strip().upper().removesuffix(".NS") for s in symbols))
    if not symbols or any(not s or '.' in s for s in symbols):
        raise ValueError("Supply NSE equity symbols (for example RELIANCE)")
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    frames = []
    for symbol in symbols:
        raw = downloader(symbol + ".NS", start=start, end=end, interval="1d",
                         auto_adjust=True, actions=False, progress=False,
                         threads=False, multi_level_index=False, timeout=30)
        if raw is None or raw.empty:
            raise ValueError(f"No historical data for {symbol}; cache was not updated")
        raw = raw.copy()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        dates = pd.DatetimeIndex(raw.index)
        if dates.tz is not None:
            dates = dates.tz_convert("Asia/Kolkata").tz_localize(None)
        raw.columns = raw.columns.str.lower()
        raw["date"] = dates.normalize()
        raw["symbol"] = symbol
        frame = validate_daily(raw)
        frame = frame[(frame.date >= pd.Timestamp(start)) & (frame.date < pd.Timestamp(end))]
        if frame.empty:
            raise ValueError(f"No data within requested interval for {symbol}")
        frames.append(frame)
    result = validate_daily(pd.concat(frames, ignore_index=True))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps({
        "source": "Yahoo Finance via yfinance", "symbols": symbols,
        "start_inclusive": str(start), "end_exclusive": str(end),
        "adjustment": "auto_adjust=True (provider-adjusted OHLC)",
        "rows": len(result), "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="data/local/daily.csv")
    args = parser.parse_args()
    data = download_nse(args.symbols, args.start, args.end, args.output)
    print(f"Saved {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
