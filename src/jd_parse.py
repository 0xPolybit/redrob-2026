"""OFFLINE: decompose the JD into a machine-readable rubric.

Reads job_description.docx, asks Claude to translate the adversarial prose into
structured positive signals / disqualifiers / anti-signals / soft preferences,
plus a handful of short "requirement probe" sentences we embed later so we can
measure each candidate's semantic fit to each must-have.

Writes artifacts/jd_rubric.json. Allowed because it runs at build time, once.
"""
from __future__ import annotations

import argparse
import json

from . import config
from .llm import docx_to_text, get_client, structured_call

JD_DOCX = config.ROOT / "job_description.docx"

# Deterministic, LLM-free rubric — lets the pipeline build before an API key is
# available. Faithful to job_description.docx; replaced by the LLM version when
# `python -m src.jd_parse` (no --fallback) runs.
_FALLBACK_RUBRIC = {
    "role_title": "Senior AI Engineer — Founding Team",
    "must_haves": [
        "Production experience with embeddings-based retrieval deployed to real users (embedding drift, index refresh, retrieval-quality regression).",
        "Production vector database or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch).",
        "Strong Python and code quality.",
        "Hands-on evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B interpretation).",
        "Shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.",
    ],
    "disqualifiers": [
        "Pure-research career (academic/research-only roles) with no production deployment.",
        "AI experience is only recent (<12 months) LangChain-calls-OpenAI with no pre-LLM ML production.",
        "Senior who hasn't written production code in 18+ months due to architecture/tech-lead drift.",
        "Entire career at services/consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) with no product-company stint.",
        "Primary expertise in computer vision, speech, or robotics without NLP/IR exposure.",
    ],
    "anti_signals": [
        "Keyword stuffer: skills list loaded with AI terms but titles/descriptions and assessment scores don't back them up.",
        "Plain-language strong candidate who built a recommendation/search system at a product company without ever writing 'RAG' or 'Pinecone' — reward these.",
        "Behavioral twins: near-identical profiles separated only by engagement signals.",
        "Honeypot: logically impossible profile (tenure > company age, expert in many skills with 0 years used, experience math that doesn't add up).",
    ],
    "soft_preferences": [
        "Located in or willing to relocate to Noida or Pune; Hyderabad/Mumbai/Delhi NCR welcome.",
        "6-8 years total, 4-5 in applied ML at product companies.",
        "Active on the platform recently so they're reachable.",
        "Notice period <= 30 days preferred.",
    ],
    "requirement_probes": [
        "I built and shipped a production embeddings-based retrieval system to real users and handled embedding drift, index refresh, and retrieval-quality regressions.",
        "I ran a vector database or hybrid search infrastructure in production, such as Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, or Elasticsearch.",
        "I designed evaluation frameworks for ranking systems using NDCG, MRR, and MAP and interpreted offline-to-online correlation and A/B tests.",
        "I shipped an end-to-end ranking, search, or recommendation system to real users at meaningful scale at a product company.",
        "I write strong production Python and have years of applied machine-learning experience predating the recent LLM hype.",
    ],
}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role_title": {"type": "string"},
        "must_haves": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hard positive signals (production embeddings/retrieval, vector DB, eval literacy, shipped a ranking/search/rec system, strong Python).",
        },
        "disqualifiers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hard negatives that push a candidate down hard.",
        },
        "anti_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Dataset traps: keyword stuffers, plain-language strong candidates, behavioral twins, honeypots.",
        },
        "soft_preferences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tie-breakers, not gates (location, notice period, years band, platform activity).",
        },
        "requirement_probes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-7 short first-person sentences a perfect candidate would truthfully say, one per core must-have, for embedding.",
            "minItems": 4,
            "maxItems": 7,
        },
    },
    "required": [
        "role_title",
        "must_haves",
        "disqualifiers",
        "anti_signals",
        "soft_preferences",
        "requirement_probes",
    ],
}

_SYSTEM = (
    "You are an expert technical recruiter and ML hiring manager. You translate "
    "a deliberately adversarial job description into a precise, machine-readable "
    "rubric. The right answer is NOT 'most AI keywords' — reason about what the "
    "JD means, not just what it says. Be concrete and faithful to the JD text."
)

_USER_TEMPLATE = (
    "Decompose the following Senior AI Engineer job description into a structured "
    "rubric. The requirement_probes must be short, plain first-person sentences "
    "(e.g. 'I built and shipped a production hybrid search system serving real "
    "users and handled index refresh and retrieval-quality regressions.') — one "
    "for each core must-have, suitable for sentence-embedding similarity against "
    "candidate profiles.\n\n=== JOB DESCRIPTION ===\n{jd}\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="write the deterministic rubric without calling the LLM (no API key needed)",
    )
    args = ap.parse_args()

    if args.fallback:
        rubric = _FALLBACK_RUBRIC
        print("Using deterministic fallback rubric (no LLM call).")
    else:
        jd_text = docx_to_text(JD_DOCX)
        client = get_client()
        rubric = structured_call(
            client,
            system=_SYSTEM,
            user=_USER_TEMPLATE.format(jd=jd_text),
            schema=_SCHEMA,
            max_tokens=4096,
        )
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.JD_RUBRIC.write_text(json.dumps(rubric, indent=2), encoding="utf-8")
    print(f"Wrote {config.JD_RUBRIC}")
    print(f"  must_haves={len(rubric['must_haves'])} "
          f"disqualifiers={len(rubric['disqualifiers'])} "
          f"probes={len(rubric['requirement_probes'])}")


if __name__ == "__main__":
    main()
