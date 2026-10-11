"""LightGBM LambdaRank with each session's stock cross-section as a query."""
import numpy as np
from lightgbm import LGBMRanker
from features.technical_indicators import FEATURES, TARGET


def fit_ranker(training, *, seed=42, n_estimators=100):
    training = training.dropna(subset=FEATURES + [TARGET]).sort_values(["date", "symbol"]).copy()
    sizes = training.groupby("date").symbol.transform("size")
    training = training[sizes >= 2]
    if training.empty:
        raise ValueError("Ranking requires at least two eligible stocks per training session")
    # Equal returns get equal relevance; fixed linear gains avoid exponential gain bias.
    percentiles = training.groupby("date")[TARGET].rank(method="average", pct=True)
    relevance = np.minimum((percentiles * 5).astype(int), 4)
    model = LGBMRanker(objective="lambdarank", label_gain=[0, 1, 2, 3, 4],
                       n_estimators=n_estimators, learning_rate=.05, num_leaves=15,
                       min_child_samples=10, random_state=seed, n_jobs=1, verbosity=-1)
    model.fit(training[FEATURES], relevance, group=training.groupby("date", sort=True).size().to_numpy())
    return model


def rank_stocks(model, frame):
    result = frame.dropna(subset=FEATURES).copy()
    if result.empty:
        return result.assign(score=np.nan, rank=np.nan)
    result["score"] = model.predict(result[FEATURES])
    result = result.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    result["rank"] = result.groupby("date").cumcount() + 1
    return result
