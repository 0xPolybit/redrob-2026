# Redrob Ranker

Hybrid retrieval (dense + BM25) → engineered features → LightGBM LambdaRank trained on
LLM-generated silver labels. All LLM/network work is **offline**; `rank.py` is CPU-only,
no-network, and finishes within the 5 min / 16 GB budget.

See `PLAN.md` for the full architecture and rationale.

## Setup
```
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# place the data at data/candidates.jsonl (100000 lines)
```

## Build artifacts (offline, may exceed 5 min — network + LLM allowed here)
The offline scripts use the Anthropic SDK and need `ANTHROPIC_API_KEY` set. Run
them as modules (relative imports):
```
python -m src.jd_parse        # JD -> artifacts/jd_rubric.json (+ requirement probes)
python -m src.embed           # -> embeddings.npy, faiss.index, bm25.pkl, ids.npy, jd_vectors.npy, features.parquet
python -m src.silver_labels   # LLM grades a sample -> artifacts/silver_labels.parquet
python -m src.train_ranker    # -> artifacts/ranker.txt (+ ranker_features.json)
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv  # first pass -> artifacts/top100.json
python -m src.reasoning       # grounded reasoning for the top 100 -> artifacts/reasoning.parquet
```

## Produce the submission (the timed step organizers reproduce — ≤5 min, CPU, NO network)
```
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

## Validate before uploading
```
bash scripts/validate.sh ./submission.csv
```

## Reproduce in Docker (matches Stage-3 environment)
```
docker build -t redrob-ranker .
docker run --rm --network none -m 16g -v "$PWD/data:/app/data" \
  redrob-ranker python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```
