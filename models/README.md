# Swing ranking model using the saved indicators

Train and score from the repository root in VS Code or a Python terminal:

```sh
python -m pip install -r requirements.txt
python -m models.train_swing_ranker
python -m models.predict_swing --feature-file features/derived/features_2022.csv.xz --date 2022-12-30 --top-k 10
python -m pytest -q
```

Training reads the four saved yearly feature files offline. All ten indicator families contribute to 19 input columns. EMA, MACD, ATR, Bollinger position, and Supertrend are expressed as price-relative ratios; absolute stock prices, dates, symbols, and future outcomes are excluded from the model matrix. There is no learned normalization fitted on future rows. `models/swing_dataset.py` defines the exact ordered input schema.

The target is the gross open-to-open return: a signal at close t enters at open t+1 and exits at open t+6, five session intervals later. Session offsets use the official index calendar. Every entry-through-exit daily bar must exist, have consistent positive OHLC, and nonzero volume; suspension/missing bars invalidate the label rather than jumping to another observed day. Labels are computed separately from the saved feature files, which remain unchanged. No fees are deducted from training labels.

LightGBM LambdaRank uses each date's stock cross-section as a query. Forward-return percentiles produce five relevance levels with linear gains; tied returns receive equal relevance. Scores represent relative ordering, not predicted returns or buy probabilities. Symbol order deterministically breaks equal prediction scores.

## Chronological selection and test

| Stage | Data | Purpose |
|---|---|---|
| Initial training | 2019–2020; 154,516 labelled rows | Learn ranking trees |
| Validation | 2021; 80,960 labelled rows | Select tree count via NDCG@10 and early stopping |
| Saved-model refit | Selected 2019–2021 rows; 235,476 rows | Fit 81 trees at the chosen count |
| Held-out test | 2022; 88,578 scored rows, 86,325 known outcomes | Evaluate once, without tuning on test results |

At each split, a training/validation label must finish within its own year boundary and strictly before the next split's first signal. Purged rows are not silently restored for the final refit. Parameters are fixed at 15 leaves, learning rate 0.03, minimum 100 child samples, L2 regularization 1, seed 42, and two threads. At most 300 trees are considered, with 30-round early stopping on 2021 only. Dates, horizon, iterations, and output directory can be changed using CLI flags; changing them creates a different research experiment.

The held-out mean daily rank IC is **0.0168**, and NDCG@10 is **0.4955**. Mean top-10 gross five-session return is **0.0942%**, compared with **0.1399%** for the eligible universe and **0.6624%** for a simple 20-session relative-strength ranking on the same sessions. These results are weak; this model does not beat that baseline. Results are preserved without test-year optimization.

IC and NDCG use stocks with known outcomes. Scoring and saved ranks include all stocks with complete inputs, including those without a known future outcome. Top-10 mean returns require all ten selected outcomes to be known; 242 test sessions qualify, and the six final sessions have no completed labels. Daily five-session observations overlap: these means are not a compounded portfolio return, annualized return, or cost-aware backtest. Daily metrics and coverage counts make the evaluation population visible.

## Saved artifacts

`models/trained/swing_ranker/` contains:

- `ranker.txt`: native LightGBM text model, trained only through 2021 outcomes. Load using `lightgbm.Booster(model_file=...)`.
- `report.json`: ordered feature schema, training/validation/test coverage, metrics, baseline, parameters, dependency versions, input hashes, and model hash.
- `test_predictions.csv.xz`: all 2022 research scores/ranks, entry/exit dates, raw prices, and realized labels where available.
- `test_daily_metrics.csv`: per-session held-out metrics and outcome coverage.
- `feature_importance.csv`: gain and split counts; these are model usage measures, not causal effects.

Saving and reloading the native model must preserve predictions. The inference runner checks feature order and rejects dates overlapping the saved model's training outcomes. It returns research rankings only and never places orders. The final available session in the existing feature files is December 2022; these predictions are historical, not current recommendations.

Prices and labels remain unadjusted. The test includes 71 labelled stock-days with absolute five-session returns over 50%, which can include corporate-action artifacts. The current constituent snapshot adds survivorship bias, and missing historical symbols are not reconstructed. Corporate-action reconciliation, historical membership, liquidity constraints, and cost-aware walk-forward portfolio validation remain necessary before interpreting this as a deployable strategy. Live trading remains disabled. The earlier baseline `models/ranking.py` and walk-forward runner retain their existing feature set.

API reference: [LightGBM LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).
