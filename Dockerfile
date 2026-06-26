# CPU image matching the Stage-3 reproduction environment.
# The timed step runs with NO network:
#   docker run --rm --network none -m 16g -v "$PWD/data:/app/data" \
#     redrob-ranker python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=8

WORKDIR /app

# Install deps first for layer caching. libgomp1 is needed by LightGBM/FAISS.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source + precomputed artifacts (artifacts/ must be built before docker build).
COPY src/ ./src/
COPY eval/ ./eval/
COPY artifacts/ ./artifacts/
COPY validate_submission.py ./

CMD ["python", "src/rank.py", "--candidates", "./data/candidates.jsonl", "--out", "./submission.csv"]
