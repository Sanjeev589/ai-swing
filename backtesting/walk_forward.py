"""Expanding-window ranking and non-overlapping next-open research portfolios."""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data.nse_loader import load_daily_csv, validate_daily
from features.technical_indicators import FEATURES, TARGET, build_features
from models.ranking import fit_ranker, rank_stocks


@dataclass(frozen=True)
class WalkForwardConfig:
    horizon: int = 5
    min_train_sessions: int = 252
    test_sessions: int = 63
    top_k: int = 10
    max_weight: float = .10
    cost_bps_per_side: float = 10
    seed: int = 42
    n_estimators: int = 100

    def __post_init__(self):
        for name in ("horizon", "min_train_sessions", "test_sessions", "top_k", "n_estimators"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.max_weight) or not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not np.isfinite(self.cost_bps_per_side) or not 0 <= self.cost_bps_per_side < 10000:
            raise ValueError("Invalid per-side cost")


def walk_forward_predictions(features, config):
    eligible = features.dropna(subset=FEATURES).copy()
    dates = pd.DatetimeIndex(sorted(eligible.date.unique()))
    outputs, folds = [], []
    first_offset = next((i for i in range(config.min_train_sessions, len(dates))
                         if eligible[(eligible.date < dates[i]) &
                                     (eligible.label_end_date < dates[i])].dropna(subset=[TARGET]).date.nunique()
                         >= config.min_train_sessions), len(dates))
    for offset in range(first_offset, len(dates), config.test_sessions):
        test_dates = dates[offset:offset + config.test_sessions]
        start = test_dates[0]
        train = eligible[(eligible.date < start) & (eligible.label_end_date < start)].dropna(subset=[TARGET])
        train_dates = train.date.nunique()
        if train_dates < config.min_train_sessions:
            continue
        model = fit_ranker(train, seed=config.seed, n_estimators=config.n_estimators)
        predicted = rank_stocks(model, eligible[eligible.date.isin(test_dates)])
        predicted["fold"] = len(folds)
        outputs.append(predicted)
        folds.append({"fold": len(folds), "train_start": train.date.min(),
                      "train_end": train.date.max(), "latest_training_label": train.label_end_date.max(),
                      "test_start": start, "test_end": test_dates[-1], "train_rows": len(train)})
    if not outputs:
        raise ValueError("Insufficient history for a purged walk-forward fold")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(folds)


def portfolio_backtest(prices, predictions, config):
    prices = validate_daily(prices)
    opens = prices.pivot(index="date", columns="symbol", values="open").sort_index()
    # Fixed-universe baseline: refuse to silently forward-fill suspensions/delistings.
    if opens.isna().any().any():
        raise ValueError("Backtest requires a complete session panel; missing prices need an explicit delisting policy")
    if predictions.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate predictions")
    if predictions.empty or not np.isfinite(predictions.score).all():
        raise ValueError("Expected finite out-of-sample scores")
    if not predictions.symbol.isin(opens.columns).all() or not predictions.date.isin(opens.index).all():
        raise ValueError("Predictions must match price symbols and sessions")
    calendar = opens.index
    first_signal = calendar.get_loc(predictions.date.min())
    cash, holdings, equity_rows, trades = 1.0, {}, [], []
    exit_index = None
    next_signal = first_signal
    cost = config.cost_bps_per_side / 10000
    for index in range(first_signal + 1, len(calendar)):
        date = calendar[index]
        if exit_index == index:
            for symbol, position in holdings.items():
                proceeds = position["shares"] * opens.loc[date, symbol] * (1 - cost)
                cash += proceeds
                trades.append({"symbol": symbol, "signal_date": position["signal_date"],
                               "entry_date": position["entry_date"], "exit_date": date,
                               "entry_price": position["entry_price"], "exit_price": opens.loc[date, symbol],
                               "weight": position["weight"],
                               "net_return": proceeds / position["allocated"] - 1})
            holdings = {}
            exit_index = None
        if index - 1 == next_signal:
            signal_date = calendar[next_signal]
            candidates = predictions[predictions.date.eq(signal_date)].sort_values(
                ["score", "symbol"], ascending=[False, True]).head(config.top_k)
            next_signal += config.horizon
            if index + config.horizon < len(calendar) and not candidates.empty:
                capital = cash
                weight = min(1 / len(candidates), config.max_weight)
                for symbol in candidates.symbol:
                    allocated = capital * weight
                    entry = opens.loc[date, symbol]
                    shares = allocated / (entry * (1 + cost))
                    cash -= allocated
                    holdings[symbol] = dict(shares=shares, allocated=allocated, weight=weight,
                                            entry_date=date, entry_price=entry, signal_date=signal_date)
                exit_index = index + config.horizon
        equity = cash + sum(p["shares"] * opens.loc[date, s] for s, p in holdings.items())
        equity_rows.append({"date": date, "equity": equity, "cash": cash,
                            "positions": len(holdings)})
    curve = pd.DataFrame(equity_rows)
    if curve.empty or not trades:
        raise ValueError("No completed out-of-sample trades; add history or reduce the horizon")
    values = np.r_[1., curve.equity.to_numpy()]
    returns = values[1:] / values[:-1] - 1
    drawdown = values / np.maximum.accumulate(values) - 1
    volatility = returns.std(ddof=1) if len(returns) > 1 else 0
    summary = {"total_return": float(values[-1] - 1),
               "max_drawdown": float(drawdown.min()), "sessions": len(curve),
               "sharpe_252": float(np.sqrt(252) * returns.mean() / volatility) if volatility > 0 else None,
               "completed_trades": len(trades), "live_trading": False}
    curve["daily_return"] = returns
    return curve, pd.DataFrame(trades), summary


def run_walk_forward(csv_path, config=None):
    config = config or WalkForwardConfig()
    prices = load_daily_csv(csv_path)
    predictions, folds = walk_forward_predictions(build_features(prices, config.horizon), config)
    curve, trades, summary = portfolio_backtest(prices, predictions, config)
    scored = predictions.dropna(subset=[TARGET])
    correlations = []
    for _, group in scored.groupby("date"):
        if len(group) > 1 and group.score.nunique() > 1 and group[TARGET].nunique() > 1:
            correlations.append(group.score.rank().corr(group[TARGET].rank()))
    summary["mean_rank_ic"] = float(np.mean(correlations)) if correlations else None
    summary["rank_ic_sessions"] = len(correlations)
    return {"predictions": predictions, "folds": folds, "equity": curve,
            "trades": trades, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="artifacts/walk_forward")
    parser.add_argument("--min-train-sessions", type=int, default=252)
    parser.add_argument("--test-sessions", type=int, default=63)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-weight", type=float, default=.10)
    parser.add_argument("--cost-bps-per-side", type=float, default=10)
    args = parser.parse_args()
    config = WalkForwardConfig(**{k: v for k, v in vars(args).items() if k not in {"csv", "output"}})
    result = run_walk_forward(args.csv, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "folds", "equity", "trades"):
        result[name].to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
