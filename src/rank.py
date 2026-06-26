"""ONLINE entrypoint — produces submission.csv. The TIMED step.

Constraints it must honor: ≤5 min wall-clock, ≤16 GB RAM, CPU only, NO network.
It loads only precomputed artifacts and a tiny LightGBM model; it imports no
LLM/embedding/network libraries and downloads nothing.

Pipeline (PLAN.md §4):
  A hybrid retrieval (dense kNN ∪ BM25, RRF) -> ~2000 shortlist
  B gather precomputed feature rows for the shortlist
  C LightGBM scores; blend with the transparent rubric
  D rule layer: hard-demote honeypots/disqualifiers; behavioral availability multiplier
  E top 100, deterministic strictly-decreasing scores, attach reasoning -> CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

# Defensive: guarantee no model/hub fetch even if some lib is transitively imported.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

# Support both `python src/rank.py` and `python -m src.rank`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config, features
    from src.build_text import tokenize
else:
    from . import config, features
    from .build_text import tokenize

TOP_N = 100

_DISQ_PENALTY = {
    "disq_research_only": 0.30,
    "disq_consulting_only": 0.35,
    "disq_cv_speech_robotics_primary": 0.40,
    "disq_recent_langchain_only": 0.60,
    "disq_architect_drift": 0.75,
}


def _rrf_fuse(rank_lists: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _retrieve(emb, jd_vecs, bm25, query_tokens) -> list[int]:
    """Stage A: dense kNN (pooled + each probe) ∪ BM25, fused via RRF."""
    import faiss

    index = faiss.read_index(str(config.FAISS_INDEX))
    # dense: one search per JD vector (pooled is row 0), union via RRF
    _, dense_ids = index.search(jd_vecs, config.DENSE_TOPK)
    dense_lists = [row.tolist() for row in dense_ids]

    # sparse: BM25 scores over all candidates, take top-k
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = np.argsort(-bm25_scores)[: config.BM25_TOPK].tolist()

    fused = _rrf_fuse(dense_lists + [bm25_top])
    shortlist = sorted(fused, key=lambda i: -fused[i])[: config.SHORTLIST_SIZE]
    return shortlist


def _load_feature_rows(ids: np.ndarray) -> list[dict]:
    """Feature rows (dicts) aligned with the embeddings/ids order."""
    import polars as pl

    df = pl.read_parquet(config.FEATURES_PARQUET)
    order = {cid: i for i, cid in enumerate(df["candidate_id"].to_list())}
    perm = [order[c] for c in ids.tolist()]
    return df[perm].to_dicts()


def _template_reasoning(row: dict) -> str:
    """Zero-hallucination placeholder reasoning from real feature numbers.

    Used only until reasoning.py writes grounded LLM reasoning; every number
    here comes straight from the candidate's computed features.
    """
    return (
        f"{row['years_of_experience']:.1f} yrs experience, "
        f"~{row['est_applied_ml_years']:.1f} yrs applied ML; "
        f"{int(row['num_core_ai_skills'])} core retrieval/ranking skills; "
        f"product-company ratio {row['product_ratio']:.2f}; "
        f"recruiter response {row['recruiter_response_rate']:.2f}."
    )


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Produce the ranked submission CSV.")
    ap.add_argument("--candidates", default=None, help="path to candidates.jsonl")
    ap.add_argument("--out", default=str(config.ROOT / "submission.csv"))
    args = ap.parse_args()

    cand_path = config.resolve_candidates_path(args.candidates)
    if not cand_path.exists():
        raise SystemExit(f"candidates file not found: {cand_path}")

    # --- load artifacts ---
    ids = np.load(config.IDS_NPY, allow_pickle=True)
    emb = np.load(config.EMBEDDINGS, mmap_mode="r")
    jd_vecs = np.load(config.JD_VECTORS).astype(np.float32)
    with open(config.BM25_PKL, "rb") as fh:
        bm25 = pickle.load(fh)

    # The LightGBM model is optional: without it we ship the transparent rubric
    # scorer (the insurance-policy baseline).
    model = None
    feat_cols: list[str] = []
    if config.RANKER_TXT.exists() and config.RANKER_FEATURES.exists():
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(config.RANKER_TXT))
        feat_cols = json.loads(config.RANKER_FEATURES.read_text(encoding="utf-8"))

    rubric = json.loads(config.JD_RUBRIC.read_text(encoding="utf-8"))
    query_text = " ".join(rubric.get("must_haves", []) + rubric.get("requirement_probes", []))
    query_tokens = tokenize(query_text)

    rows = _load_feature_rows(ids)
    mode = "LightGBM⊕rubric blend" if model is not None else "rubric-only (no model)"
    print(f"[{time.time()-t0:.1f}s] artifacts loaded ({len(ids)} candidates) — {mode}")

    # --- Stage A: retrieval ---
    shortlist = _retrieve(np.asarray(emb[:]), jd_vecs, bm25, query_tokens)
    print(f"[{time.time()-t0:.1f}s] shortlist = {len(shortlist)}")

    # --- Stage B + C: features -> model score blended with rubric ---
    rubric_scores = np.array([features.rubric_score(rows[i]) for i in shortlist])
    if model is not None:
        Xs = np.array([[rows[i][c] for c in feat_cols] for i in shortlist], dtype=np.float32)
        model_raw = model.predict(Xs)
        rng = model_raw.max() - model_raw.min()
        model_norm = (model_raw - model_raw.min()) / (rng + 1e-9)
        blend = config.LTR_BLEND * model_norm + (1 - config.LTR_BLEND) * rubric_scores
    else:
        blend = rubric_scores

    # --- Stage D: rule layer ---
    final = np.empty(len(shortlist), dtype=np.float64)
    for j, idx in enumerate(shortlist):
        r = rows[idx]
        s = float(blend[j])
        if r.get("honeypot", 0) >= 1:
            final[j] = -1.0  # impossibility -> floor; never reaches top 100
            continue
        for flag, pen in _DISQ_PENALTY.items():
            if r.get(flag, 0) >= 1:
                s *= pen
        s *= r.get("availability_mult", 1.0)
        final[j] = s

    # --- Stage E: top 100, deterministic strictly-decreasing scores ---
    order = sorted(
        range(len(shortlist)),
        key=lambda j: (-final[j], ids[shortlist[j]]),
    )[:TOP_N]

    reasoning_map = {}
    if config.REASONING_PARQUET.exists():
        import polars as pl

        rp = pl.read_parquet(config.REASONING_PARQUET)
        reasoning_map = dict(zip(rp["candidate_id"].to_list(), rp["reasoning"].to_list()))

    out_rows = []
    prev = float("inf")
    top_ids = []
    for rank, j in enumerate(order, start=1):
        idx = shortlist[j]
        cid = str(ids[idx])
        top_ids.append(cid)
        score = round(float(final[j]), 6)
        if score >= prev:                 # enforce strictly decreasing (no ties to break)
            score = round(prev - 1e-6, 6)
        prev = score
        reasoning = reasoning_map.get(cid) or _template_reasoning(rows[idx])
        out_rows.append((cid, rank, score, reasoning))

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in out_rows:
            w.writerow([cid, rank, f"{score:.6f}", reasoning])

    # persist the top-100 ids so reasoning.py can generate grounded reasoning
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "top100.json").write_text(json.dumps(top_ids), encoding="utf-8")

    hp = sum(1 for j in order if rows[shortlist[j]].get("honeypot", 0) >= 1)
    print(f"[{time.time()-t0:.1f}s] wrote {args.out} (honeypots in top 100: {hp})")


if __name__ == "__main__":
    main()
