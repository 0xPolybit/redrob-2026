"""OFFLINE: train the LightGBM LambdaRank model on the silver labels.

LambdaRank optimizes NDCG directly, aligning with the competition's composite.
There is a single query (one JD), so the whole sample is one ranking group; we
hold out a slice for validation and compare the model against the transparent
rubric baseline using the local metrics. Writes artifacts/ranker.txt and the
exact feature-column list (artifacts/ranker_features.json) rank.py must reuse.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, features
from eval import metrics


def _model_feature_columns(df) -> list[str]:
    skip = features.NON_MODEL_COLUMNS | {"reason", "tier"}
    cols = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in skip:
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def main() -> None:
    import lightgbm as lgb
    import polars as pl

    feats = pl.read_parquet(config.FEATURES_PARQUET)
    labels = pl.read_parquet(config.SILVER_LABELS).select(["candidate_id", "tier"])
    df = feats.join(labels, on="candidate_id", how="inner")
    print(f"Training rows (labeled ∩ featured): {df.height}")
    if df.height < 50:
        raise SystemExit("Too few labeled rows to train; run silver_labels.py first.")

    feat_cols = _model_feature_columns(df)
    config.RANKER_FEATURES.write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    print(f"{len(feat_cols)} model features")

    # deterministic train/valid split (single ranking group each)
    rng = np.random.default_rng(7)
    perm = rng.permutation(df.height)
    n_valid = max(50, df.height // 5)
    valid_idx, train_idx = perm[:n_valid], perm[n_valid:]

    X = df.select(feat_cols).to_numpy()
    y = df["tier"].to_numpy()
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[valid_idx], y[valid_idx]

    train_set = lgb.Dataset(Xtr, label=ytr, group=[len(Xtr)])
    valid_set = lgb.Dataset(Xva, label=yva, group=[len(Xva)], reference=train_set)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "label_gain": [2**i - 1 for i in range(6)],  # tiers 0..5
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 7,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )
    model.save_model(str(config.RANKER_TXT))
    print(f"Wrote {config.RANKER_TXT}")

    # --- compare model vs transparent rubric on the held-out slice ---
    valid_rows = df[valid_idx.tolist()].to_dicts()
    model_scores = model.predict(Xva)
    order_model = np.argsort(-model_scores)
    rubric_scores = np.array([features.rubric_score(r) for r in valid_rows])
    order_rubric = np.argsort(-rubric_scores)

    m_model = metrics.composite(yva[order_model])
    m_rubric = metrics.composite(yva[order_rubric])
    print("Held-out composite — LightGBM:", {k: round(v, 4) for k, v in m_model.items()})
    print("Held-out composite — rubric  :", {k: round(v, 4) for k, v in m_rubric.items()})

    # feature importances (top 15)
    imp = sorted(zip(feat_cols, model.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])[:15]
    print("Top features by gain:")
    for name, gain in imp:
        print(f"  {name}: {gain:.0f}")


if __name__ == "__main__":
    main()
