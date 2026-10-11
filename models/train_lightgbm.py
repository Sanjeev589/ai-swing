"""Chronological baseline model. No randomized time split."""
import argparse
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from data.nse_loader import load_daily_csv
from features.technical_indicators import FEATURES, build_features

def fit_and_evaluate(csv_path: str, train_end="2022-12-31", validation_end="2024-12-31", horizon=5):
    df = build_features(load_daily_csv(csv_path), horizon=horizon)
    df = df.dropna(subset=FEATURES + ["target_5d_return"]).copy()
    # Purge labels crossing period boundaries: a label uses a future price.
    df = df.dropna(subset=["label_end_date"])
    train = df[(df.date <= train_end) & (df.label_end_date <= train_end)]
    val = df[(df.date > train_end) & (df.date <= validation_end) & (df.label_end_date <= validation_end)]
    test = df[df.date > validation_end]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Insufficient data for all three chronological splits")
    model = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, random_state=42, verbosity=-1, n_jobs=1)
    model.fit(train[FEATURES], train["target_5d_return"])
    validation_prediction = model.predict(val[FEATURES])
    print(f"Validation MAE: {mean_absolute_error(val['target_5d_return'], validation_prediction):.6f}")
    prediction = model.predict(test[FEATURES])
    print(f"Train={len(train)} Validation={len(val)} Test={len(test)}")
    print(f"Test MAE: {mean_absolute_error(test['target_5d_return'], prediction):.6f}")
    output = test[["date", "symbol", "target_5d_return"]].copy()
    output["predicted_return"] = prediction
    return model, output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-end", default="2024-12-31")
    args = parser.parse_args()
    fit_and_evaluate(args.csv, args.train_end, args.validation_end)
