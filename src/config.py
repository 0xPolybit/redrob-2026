"""Central configuration: paths, model ids, and the JD-derived constants.

Everything that more than one module needs to agree on lives here so the JD
logic is defined once and reused by features.py, honeypot.py, silver_labels.py
and rank.py rather than copy-pasted.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# Generated artifacts (offline build writes these; rank.py reads them).
JD_RUBRIC = ARTIFACTS_DIR / "jd_rubric.json"
EMBEDDINGS = ARTIFACTS_DIR / "embeddings.npy"
FAISS_INDEX = ARTIFACTS_DIR / "faiss.index"
BM25_PKL = ARTIFACTS_DIR / "bm25.pkl"
IDS_NPY = ARTIFACTS_DIR / "ids.npy"            # candidate_id order, aligned with embeddings rows
JD_VECTORS = ARTIFACTS_DIR / "jd_vectors.npy"  # [pooled, req_0, req_1, ...] probe embeddings
FEATURES_PARQUET = ARTIFACTS_DIR / "features.parquet"
SILVER_LABELS = ARTIFACTS_DIR / "silver_labels.parquet"
RANKER_TXT = ARTIFACTS_DIR / "ranker.txt"
RANKER_FEATURES = ARTIFACTS_DIR / "ranker_features.json"
REASONING_PARQUET = ARTIFACTS_DIR / "reasoning.parquet"


def resolve_candidates_path(explicit: str | os.PathLike | None = None) -> Path:
    """Locate candidates.jsonl: explicit arg, then data/, then repo root."""
    if explicit:
        return Path(explicit)
    for cand in (DATA_DIR / "candidates.jsonl", ROOT / "candidates.jsonl"):
        if cand.exists():
            return cand
    # default to the documented location even if missing (caller will error clearly)
    return DATA_DIR / "candidates.jsonl"


# --- models --------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, CPU-friendly, strong on retrieval
CLAUDE_MODEL = "claude-opus-4-8"          # offline-only (jd_parse, silver_labels, reasoning)

# BGE retrieval works best with an instruction prefix on the *query* side.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- retrieval / ranking knobs ------------------------------------------
SHORTLIST_SIZE = 2000      # candidates kept after hybrid retrieval
DENSE_TOPK = 2000
BM25_TOPK = 2000
RRF_K = 60                 # reciprocal-rank-fusion constant
LTR_BLEND = 0.7            # final = LTR_BLEND*model + (1-LTR_BLEND)*rubric
SILVER_SAMPLE_SIZE = 2000  # candidates sent to the LLM for silver labels

# --- JD-derived vocabularies (from job_description.docx) -----------------
# "People who have only worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "deloitte", "ibm global services", "dxc", "genpact",
}

# "Located in or willing to relocate to Noida or Pune; Hyderabad/Mumbai/Delhi NCR welcome."
PREFERRED_LOCATIONS = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "delhi ncr", "ncr",
    "gurgaon", "gurugram", "ghaziabad", "navi mumbai",
}

# "IT Services" style industries used to detect product-vs-services career mix.
SERVICES_INDUSTRIES = {
    "it services", "information technology & services", "consulting",
    "staffing", "outsourcing", "bpo",
}

# Core must-have AI/ML signal terms (production retrieval / ranking / search).
CORE_AI_SKILLS = {
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "bge", "e5", "openai embeddings", "vector search", "vector database",
    "vector db", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "hybrid search", "retrieval", "rag",
    "learning to rank", "learning-to-rank", "ranking", "recommendation",
    "recommender", "information retrieval", "semantic search", "ndcg", "mrr",
    "bm25", "reranking", "re-ranking",
}

# Broad AI keyword set used by the keyword-stuffer detector (listed-but-unbacked).
AI_KEYWORDS = CORE_AI_SKILLS | {
    "machine learning", "deep learning", "nlp", "natural language processing",
    "llm", "llms", "large language models", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "transformers", "pytorch", "tensorflow",
    "hugging face", "huggingface", "langchain", "llamaindex", "prompt engineering",
    "generative ai", "gen ai", "mlops", "feature store", "model serving",
}

# CV/speech/robotics primary-expertise terms (a JD disqualifier without NLP/IR).
CV_SPEECH_ROBOTICS = {
    "computer vision", "image classification", "object detection", "ocr",
    "speech recognition", "tts", "text to speech", "asr", "robotics",
    "slam", "lidar", "pose estimation", "segmentation", "opencv",
}

# Research-only / academic signal terms (pure-research career is a disqualifier).
RESEARCH_TERMS = {
    "phd", "postdoc", "post-doc", "research scientist", "research fellow",
    "research assistant", "publications", "paper", "thesis", "academia",
    "university", "institute of technology", "laboratory",
}

# Recent-LangChain-only trap.
LANGCHAIN_TERMS = {"langchain", "llamaindex", "openai api", "gpt wrapper"}
