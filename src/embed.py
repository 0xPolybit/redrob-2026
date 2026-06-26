"""OFFLINE: build every precomputed artifact rank.py needs.

  candidates.jsonl ──┬─► dense embeddings (BGE-small) ─► embeddings.npy + faiss.index
                     ├─► BM25 sparse index            ─► bm25.pkl
                     ├─► candidate id order           ─► ids.npy
                     └─► deterministic + semantic features ─► features.parquet
  jd_rubric.json ────► requirement-probe embeddings   ─► jd_vectors.npy  (row 0 = pooled)

Single streaming pass over the 487 MB JSONL so we never hold the whole file as
Python objects. May exceed the 5-minute budget — that's fine, this is build time.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from . import config, features, honeypot
from .build_text import candidate_to_text, tokenize


def _iter_candidates(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _embed_texts(texts: list[str], *, prefix: str = "") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    payload = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(
        payload,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def _build_jd_vectors() -> np.ndarray:
    rubric = json.loads(config.JD_RUBRIC.read_text(encoding="utf-8"))
    probes = rubric["requirement_probes"]
    probe_vecs = _embed_texts(probes, prefix=config.BGE_QUERY_PREFIX)
    pooled = probe_vecs.mean(axis=0, keepdims=True)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9
    return np.vstack([pooled, probe_vecs]).astype(np.float32)  # row 0 pooled, 1.. probes


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cand_path = config.resolve_candidates_path()
    if not cand_path.exists():
        raise SystemExit(f"candidates file not found: {cand_path}")

    print(f"Reading {cand_path} ...")
    ids: list[str] = []
    texts: list[str] = []
    feat_rows: list[dict] = []
    n = 0
    for c in _iter_candidates(cand_path):
        cid = c["candidate_id"]
        ids.append(cid)
        texts.append(candidate_to_text(c))
        row = features.extract_features(c)
        is_hp, _ = honeypot.honeypot_flags(c)
        row["candidate_id"] = cid
        row["honeypot"] = 1.0 if is_hp else 0.0
        row["availability_mult"] = features.availability_multiplier(row)
        feat_rows.append(row)
        n += 1
        if n % 10000 == 0:
            print(f"  parsed {n} candidates")
    print(f"Total candidates: {n}")

    # --- dense embeddings + FAISS ---
    print("Embedding candidate profiles (document side) ...")
    emb = _embed_texts(texts)
    np.save(config.EMBEDDINGS, emb)
    np.save(config.IDS_NPY, np.array(ids))

    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine (vectors normalized)
    index.add(emb)
    faiss.write_index(index, str(config.FAISS_INDEX))
    print(f"  wrote {config.EMBEDDINGS.name} {emb.shape} and {config.FAISS_INDEX.name}")

    # --- JD requirement-probe vectors + semantic features for all candidates ---
    print("Embedding JD requirement probes ...")
    jd_vecs = _build_jd_vectors()
    np.save(config.JD_VECTORS, jd_vecs)
    sims = emb @ jd_vecs.T  # N x (1 + num_probes)
    sim_pooled = sims[:, 0]
    sim_probes = sims[:, 1:]
    sim_req_max = sim_probes.max(axis=1)
    sim_req_mean = sim_probes.mean(axis=1)

    # --- BM25 sparse index (free texts afterward) ---
    print("Building BM25 index ...")
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(t) for t in texts]
    del texts
    bm25 = BM25Okapi(corpus)
    with open(config.BM25_PKL, "wb") as fh:
        pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
    del corpus
    print(f"  wrote {config.BM25_PKL.name}")

    # --- assemble features.parquet (deterministic ⊕ semantic) ---
    print("Writing features.parquet ...")
    import polars as pl

    for i, row in enumerate(feat_rows):
        row["sim_pooled"] = float(sim_pooled[i])
        row["sim_req_max"] = float(sim_req_max[i])
        row["sim_req_mean"] = float(sim_req_mean[i])
    df = pl.DataFrame(feat_rows)
    # candidate_id first for readability
    cols = ["candidate_id"] + [c for c in df.columns if c != "candidate_id"]
    df.select(cols).write_parquet(config.FEATURES_PARQUET)
    print(f"  wrote {config.FEATURES_PARQUET.name} {df.shape}")
    print("embed.py done.")


if __name__ == "__main__":
    main()
