"""Local copies of the official ranking metrics (numpy only).

Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

`relevances` is the array of ground-truth relevance tiers (0..5) of the items in
the order your system ranked them — index 0 is your rank-1 pick.
"""
from __future__ import annotations

import numpy as np

RELEVANT_TIER = 3  # P@10 counts tier 3+ as "relevant" (per submission_spec)


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0**rels - 1.0) * discounts))


def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = np.asarray(rels, dtype=float)
    ideal = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def average_precision(rels: np.ndarray, relevant_tier: int = RELEVANT_TIER) -> float:
    rels = np.asarray(rels, dtype=float)
    binary = (rels >= relevant_tier).astype(float)
    total_relevant = binary.sum()
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(binary, start=1):
        if rel:
            hits += 1
            precision_sum += hits / i
    return float(precision_sum / total_relevant)


def precision_at_k(rels: np.ndarray, k: int, relevant_tier: int = RELEVANT_TIER) -> float:
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    return float(np.mean(rels >= relevant_tier))


def composite(rels: np.ndarray) -> dict[str, float]:
    """Return the individual metrics and the weighted composite."""
    rels = np.asarray(rels, dtype=float)
    m = {
        "ndcg@10": ndcg_at_k(rels, 10),
        "ndcg@50": ndcg_at_k(rels, 50),
        "map": average_precision(rels),
        "p@10": precision_at_k(rels, 10),
    }
    m["composite"] = (
        0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"] + 0.15 * m["map"] + 0.05 * m["p@10"]
    )
    return m
