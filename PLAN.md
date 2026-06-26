# Redrob Hackathon — Build Plan

**Challenge:** Intelligent Candidate Discovery & Ranking — rank the top 100 of 100,000 candidates against a single Senior AI Engineer job description.

**Chosen approach:** Hybrid retrieval (dense + sparse) → engineered features → a LightGBM learning-to-rank model trained on LLM-generated "silver" labels offline. All heavy work happens offline; the scored ranking step is a fast, CPU-only, no-network program.

This document is the team's single source of truth. Read it top to bottom before writing code.

---

## 1. What we are actually being scored on

The composite is dominated by the top of the list:

```
Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

Eighty percent of the score lives in NDCG@10 and NDCG@50. Practical consequence: **getting the top 50 right — and especially the top 10 — matters far more than the long tail.** We optimize relentlessly for precision and ordering at the head, not for breadth.

Three things can disqualify us regardless of score, so they are non-negotiable engineering constraints, not nice-to-haves:

- **Compute:** the ranking step must finish in ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU only, **no network** (no hosted LLM calls). Pre-computation may exceed 5 min; the step that emits the CSV may not.
- **Honeypots:** ~80 candidates have subtly impossible profiles and are forced to relevance tier 0 in the ground truth. If our top 100 contains > 10% honeypots we are disqualified at Stage 3. A keyword-embedding-only system walks straight into these.
- **Reproducibility + defense:** Stage 3 re-runs our `rank.py` in a clean Docker container; Stage 4 reads our git history and reasoning; Stage 5 is a 30-minute interview where we defend the architecture. The build must be real engineering with honest iteration, not a single LLM dump.

## 2. Reading the JD correctly — "beyond keywords"

The JD is deliberately adversarial. The organizers state plainly that the right answer is *not* "most AI keywords." We translate the prose JD into a structured rubric of **positive signals, disqualifiers, and anti-signals**, then score candidates against that rubric rather than against raw keywords.

**Must-haves (hard positive signals)**
- Production embeddings/retrieval experience (sentence-transformers, BGE, E5, OpenAI embeddings — model-agnostic; what matters is *production*, drift, index refresh).
- Production vector DB / hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch).
- Strong Python.
- Ranking-evaluation literacy (NDCG, MRR, MAP, A/B interpretation).
- Has shipped at least one end-to-end ranking / search / recommendation system to real users.

**Disqualifiers (hard negative — push candidate down hard)**
- Pure-research career with no production deployment.
- "AI experience" is only recent (< 12 months) LangChain-calls-OpenAI with no pre-LLM ML production.
- Senior who hasn't written production code in 18+ months ("architecture/tech-lead" drift).
- Entire career at services/consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) with no product-company stint.
- Primary expertise in CV / speech / robotics without NLP/IR.

**Anti-signals the dataset weaponizes (the traps)**
- **Keyword stuffers:** skills list is loaded with AI terms but career descriptions and titles don't back it up (e.g., title "Marketing Manager" with a perfect AI skill list). Skill-assessment scores often expose these.
- **Plain-language Tier-5s** (the *inverse* trap): a genuinely strong candidate who never writes "RAG" or "Pinecone" but whose history shows they built a recommendation system at a product company. We must *reward* these — embeddings + career reasoning catch them where keyword matching fails.
- **Behavioral twins:** near-identical profiles separated only by engagement signals.
- **Honeypots:** logically impossible profiles (tenure > company age; "expert" in 10 skills with 0 years used; experience-math that doesn't add up).

**Soft preferences (tie-breakers, not gates)**
- Located in / willing to relocate to Noida or Pune; Hyderabad/Mumbai/Delhi NCR/Pune welcome.
- 6–8 years total, 4–5 in applied ML at product companies.
- Active on the platform (so they're reachable) — this is where behavioral signals enter.
- Notice period ≤ 30 days preferred.

We encode all of the above as features. The JD itself is parsed **once, offline, with an LLM** into a machine-readable rubric (`jd_rubric.json`) — this is allowed because it happens in development, not in the ranking step.

## 3. Architecture overview

Two clean phases. The expensive, smart, network-allowed work is offline. The submission step is dumb, fast, and self-contained.

```
┌───────────────────────── OFFLINE (build time, network + LLM allowed) ─────────────────────────┐
│                                                                                                │
│  candidates.jsonl ──► profile text builder ──► dense embeddings (BGE-small / E5-small)         │
│        │                                   └──► BM25 sparse index                              │
│        │                                                                                       │
│        ├──► deterministic feature extractor ──► features.parquet (one row per candidate)       │
│        │       (career, disqualifiers, honeypot flags, behavioral signals, semantic sims)      │
│        │                                                                                       │
│  job_description ──► LLM JD decomposition ──► jd_rubric.json + JD requirement embeddings        │
│        │                                                                                       │
│        └──► sample ~2–4k candidates ──► LLM scores each vs rubric ──► silver_labels.parquet     │
│                                                          │                                      │
│                                            LightGBM LambdaRank ──► ranker.txt (model)           │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼  artifacts committed to repo
        ┌───────────────── ONLINE (rank.py — ≤5 min, 16 GB, CPU, NO network) ─────────────────┐
        │                                                                                      │
        │  load candidates.jsonl + precomputed embeddings/index + features.parquet + ranker    │
        │        │                                                                             │
        │  Stage A  hybrid retrieval: dense kNN ∪ BM25 ──► top ~2,000 candidate shortlist      │
        │  Stage B  build feature matrix for shortlist (vectorized, no per-candidate LLM)       │
        │  Stage C  LightGBM scores shortlist ──► sort                                          │
        │  Stage D  rule layer: hard-demote honeypots/disqualifiers; behavioral multiplier      │
        │  Stage E  take top 100, attach precomputed grounded reasoning ──► submission.csv      │
        │                                                                                      │
        └──────────────────────────────────────────────────────────────────────────────────┘
```

Why this shape wins:
- **It respects the constraint honestly.** No LLM-per-candidate at rank time — exactly what the spec says production systems can't afford. This is also the strongest interview answer.
- **It's a real recommender pattern** (retrieve → feature → rank → business-rule layer), which is literally the job the JD describes. The architecture *is* the cover letter.
- **Honeypots and disqualifiers are handled structurally** (rules + features), not by hoping embeddings notice.

## 4. The ranking pipeline in detail

### Stage A — Hybrid retrieval (recall)
Goal: shrink 100K → ~2,000 without dropping any true top-100 candidate.

- **Dense:** embed each candidate's `profile.summary + headline + career descriptions` with a small local model (BGE-small-en-v1.5 or e5-small-v2, ~384-dim). Embed the JD rubric (each requirement separately + a pooled JD vector). Retrieve by cosine kNN via **FAISS** (`IndexFlatIP` is fine at 100K×384 ≈ 150 MB — well within budget; HNSW if we want headroom).
- **Sparse:** **BM25** (rank-bm25 or a lightweight inverted index) over the same text to catch exact-term matches embeddings miss (specific tools, vector-DB names).
- **Fuse:** union the two candidate sets (Reciprocal Rank Fusion). Recall over precision here — the ranker fixes precision next.

Dense vs sparse is exactly the "hybrid vs dense retrieval" opinion the JD asks candidates to defend; building it gives us a real answer for Stage 5.

### Stage B — Feature engineering (the real IP)
One vectorized pass over the shortlist. Every feature is cheap and deterministic. Grouped:

*Semantic-fit features*
- Cosine(candidate, pooled JD), and cosine to each must-have requirement vector (retrieval, vector-DB, eval, shipped-ranking-system).
- Max / mean similarity of career-history descriptions to the "shipped a ranking/search/rec system" requirement.

*Career-structure features*
- Years of experience; estimated years in *applied ML* roles (from titles/descriptions).
- Product-company vs services-company ratio across career history (company names + `current_industry == "IT Services"` + size patterns).
- Recency of hands-on coding (gap since last IC/engineering role vs pure "architect/lead" titles).
- Title–skill coherence: does the current/most-recent title plausibly match the claimed AI skills? (Marketing Manager + Fine-tuning LLMs = incoherent.)

*Trap / quality features*
- **Keyword-stuffer score:** count of AI skills listed vs evidence in `career_history.description` + `redrob_signals.skill_assessment_scores`. High listed / low assessed / no description support → penalty.
- **Honeypot flags (hard):** tenure_months > (today − company_founding proxy); `proficiency == "expert"` with `duration_months == 0`; sum of role durations inconsistent with `years_of_experience`; impossible date ranges. Any hard flag ⇒ force to the bottom. (We don't special-case the 80; we build general impossibility checks, exactly as the spec recommends.)
- Disqualifier flags: pure-research, consulting-only career, recent-LangChain-only, CV/speech/robotics-primary.

*Behavioral signals (the multiplier, per `redrob_signals_doc.md`)*
- Recency: days since `last_active_date` (decay — 6 months stale ⇒ strong down-weight).
- `recruiter_response_rate`, `interview_completion_rate`, `open_to_work_flag`, `offer_acceptance_rate`, `profile_completeness_score`, verification flags.
- These combine into an **availability multiplier** applied after the model score: a perfect-on-paper but unreachable candidate (no login 6 months, 5% response) is, for hiring, not available — down-weight, don't drop.
- Location/notice tie-breakers (Noida/Pune/relocate; notice ≤ 30d).

### Stage C — Learning-to-rank model
- **Model:** LightGBM `LambdaRank` (`objective="lambdarank"`, optimizes NDCG directly — aligns with our scoring metric). Tiny, fast, CPU-native, milliseconds to score 2,000 rows.
- **Training labels (the hard part — ground truth is hidden):** we generate **silver labels** offline. Sample ~2,000–4,000 candidates spanning the score range, and have an LLM grade each against `jd_rubric.json` on the same 0–4 relevance-tier scale the organizers use (Tier 0 = honeypot/irrelevant … Tier 5 = ideal). Include deliberately-seeded honeypots and keyword-stuffers in the sample so the model learns to push them down. This is allowed: it's offline development, not the ranking step.
- **Guardrails against silver-label noise:** blend the learned score with the transparent rubric score (see below) so we never fully trust the LLM labels; validate label quality by hand on ~50 rows; use cross-validation and watch for the model simply re-learning "more keywords = better."
- **Fallback:** if silver labels prove too noisy, the same features feed a hand-weighted transparent scorer — we keep it as both a sanity baseline and an insurance policy. Ship whichever validates better; likely a blend (e.g., `0.7·LTR + 0.3·rubric`).

### Stage D — Rule layer (safety + business logic)
Applied after the model score, deterministically:
1. Hard honeypot/impossibility flags ⇒ sent to the bottom (protects the >10% disqualifier).
2. Hard disqualifiers ⇒ heavy multiplicative penalty.
3. Behavioral availability multiplier ⇒ scales the score.
4. Deterministic tie-break: model secondary score, then `candidate_id` ascending (spec requires unique ranks + non-increasing score).

### Stage E — Reasoning generation
The `reasoning` column is graded at Stage 4 for specificity, JD-connection, honest concerns, and **zero hallucination**.
- Generate reasoning **offline for the final top 100** (it's not part of the timed ranking step). Two safe options:
  - **Grounded LLM:** feed the LLM only the candidate's actual fields + their feature values + assigned rank, instruct it to cite specific facts and acknowledge concerns. Then run an automated **grounding check**: every named skill/employer/number in the reasoning must appear in the candidate's record, else regenerate.
  - **Template-with-facts fallback:** deterministic sentences assembled from real feature values (years, current title, top matched skills, one concern). Less elegant, zero hallucination risk.
- Reasoning tone must match rank (rank-5 confident, rank-95 hedged) — we feed the rank in so it's consistent.

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11** | Spec assumes it; team knows it; JD demands strong Python. |
| Data handling | **polars** (or pandas) + **pyarrow/parquet** | 465 MB uncompressed JSONL; polars is fast and memory-lean for the 16 GB cap. Parquet for precomputed features. |
| Embeddings | **sentence-transformers** with **BGE-small-en-v1.5** or **e5-small-v2** | Small, strong, CPU-friendly, ~384-dim. Runs locally → satisfies no-network. |
| Vector index | **FAISS** (`faiss-cpu`) | Industry-standard, fast kNN at 100K, trivial memory at small dim. A named JD skill. |
| Sparse retrieval | **rank-bm25** (or a hand-rolled inverted index) | Hybrid retrieval; catches exact tool/DB names. |
| Ranker | **LightGBM** (LambdaRank) | Optimizes NDCG directly, tiny/fast/CPU, great for tabular LTR. |
| Offline LLM (dev only) | Claude / GPT-4 via API | JD decomposition, silver labels, reasoning. **Never** called in `rank.py`. |
| Eval | custom NDCG@k / MAP / P@k in numpy + a held-out silver set | Mirror the official metrics locally; the JD explicitly wants eval literacy. |
| Repro | **Docker** (CPU base, pinned `requirements.txt`) | Stage-3 reproduction runs in Docker; we test in the same image. |
| Sandbox/demo | **HuggingFace Spaces** or **Streamlit Cloud** (free tier) | Mandatory Section-10.5 deliverable: upload ≤100 candidates, get ranked CSV. |
| Repo hygiene | GitHub, real commit history, `submission_metadata.yaml`, README with one repro command | Stage 4 checks for genuine iteration; Stage 5 defends it. |

Memory math (sanity): 100K × 384 float32 ≈ 150 MB embeddings; FlatIP index similar; JSONL streamed, not all held as objects; features in parquet. Comfortably under 16 GB. FAISS + LightGBM scoring of a 2K shortlist is sub-second; the 5-minute budget is mostly JSONL load — which we can pre-serialize to parquet to make `rank.py` load fast.

## 6. Repository layout

```
redrob-ranker/
├── README.md                  # one-command repro, setup, architecture summary
├── requirements.txt           # pinned deps
├── submission_metadata.yaml   # mirrors portal metadata (Section 10.2)
├── Dockerfile                 # CPU image matching the Stage-3 constraints
├── data/                      # candidates.jsonl (gitignored), JD
├── artifacts/                 # precomputed: embeddings.npy, faiss.index, bm25.pkl,
│                              #   features.parquet, jd_rubric.json, ranker.txt
├── src/
│   ├── build_text.py          # candidate → searchable/embeddable text
│   ├── jd_parse.py            # OFFLINE: JD → jd_rubric.json (+ requirement embeddings)
│   ├── embed.py               # OFFLINE: build embeddings + FAISS + BM25
│   ├── features.py            # deterministic feature extraction (shared offline/online)
│   ├── honeypot.py            # impossibility / trap detection
│   ├── silver_labels.py       # OFFLINE: LLM grades sample → silver_labels.parquet
│   ├── train_ranker.py        # OFFLINE: LightGBM LambdaRank → ranker.txt
│   ├── reasoning.py           # OFFLINE: grounded reasoning for top 100 + grounding check
│   └── rank.py                # ONLINE entrypoint: produces submission.csv (≤5 min, no net)
├── eval/
│   └── metrics.py             # NDCG@k, MAP, P@k on a local silver/held-out set
├── tests/
│   ├── test_honeypot.py       # seeded impossible profiles are caught
│   ├── test_schema.py         # output matches submission_spec section 3
│   └── test_no_network.py     # assert rank.py makes no external calls
└── scripts/
    └── validate.sh            # run organizers' validate_submission.py before upload
```

Repro contract (goes in README):
```
# offline, once (may exceed 5 min):
python src/jd_parse.py && python src/embed.py && python src/silver_labels.py && python src/train_ranker.py
# the timed step the organizers reproduce:
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

## 7. Timeline

Tune to your actual hours; this assumes a multi-day window with a mixed-skill team. Pair the less-experienced members onto features/tests (well-scoped, high-learning) and the strongest on the ranker + silver labels.

**Phase 0 — Foundation (day 1).** Load data with polars, confirm 100K, build `build_text.py`, write `metrics.py`, set up repo + git discipline (commit often — Stage 4 reads history). Deliverable: a trivial baseline (BM25-only top 100) that passes `validate_submission.py`. *Always have a valid submission in hand.*

**Phase 1 — Retrieval + features (days 1–2).** `embed.py` (embeddings + FAISS + BM25), `features.py`, `honeypot.py` with `test_honeypot.py`. Deliverable: hybrid-retrieval + transparent rubric scorer end-to-end. This alone is a respectable submission and our insurance policy.

**Phase 2 — JD decomposition + silver labels + LTR (days 2–3).** `jd_parse.py`, `silver_labels.py` (hand-validate ~50 labels), `train_ranker.py`. Compare LTR vs rubric vs blend on the held-out silver set; pick the winner. Deliverable: the real ranker.

**Phase 3 — Reasoning + rules + hardening (day 3–4).** `reasoning.py` with grounding check, rule layer, behavioral multiplier tuning. Verify honeypot rate in top 100 is ~0. Deliverable: full submission with reasoning.

**Phase 4 — Reproducibility + sandbox + submit (final day).** Build/run Docker exactly per constraints; confirm `rank.py` ≤ 5 min / ≤ 16 GB / no network on a clean machine; deploy HuggingFace/Streamlit sandbox; fill `submission_metadata.yaml`; write the 200-word methodology; run organizers' validator; submit. Keep 1 of 3 submission slots in reserve.

## 8. Risks & mitigations

- **Silver labels are noisy / the LLM also chases keywords.** → Hand-validate a sample; blend with the transparent rubric; seed honeypots+stuffers into the label set so the model learns to reject them; keep the rubric-only scorer as a shippable fallback.
- **Honeypots leak into top 100 (disqualifier).** → Dedicated `honeypot.py` with general impossibility checks + a unit test on seeded impossible profiles; assert honeypot count in final top 100 before every submission.
- **`rank.py` blows the 5-min/16 GB budget.** → Pre-serialize candidates to parquet; keep embeddings/index precomputed in `artifacts/`; vectorize features; test in the Docker image early, not the night before.
- **Accidental network call at rank time** (e.g., sentence-transformers fetching a model). → Pin/cache the model locally in the image; `test_no_network.py` blocks sockets and asserts `rank.py` still runs.
- **Over-rewarding keyword-rich profiles (the whole trap).** → Title–skill coherence + keyword-stuffer features + skill-assessment cross-check; explicitly test that a "Marketing Manager with perfect AI skills" ranks low and a "plain-language Tier-5 rec-system builder" ranks high.
- **Stage 4/5 — can't defend the work.** → Real commit history, a written design doc (this file), each member owns a module and can explain it; rehearse the hybrid-vs-dense and offline-vs-online arguments the JD itself raises.

## 9. Definition of done (pre-submission checklist)

- [ ] `validate_submission.py` passes (exactly 100 rows, unique ranks 1–100, non-increasing score, all IDs exist).
- [ ] Honeypot count in top 100 ≈ 0 (well under the 10% bar).
- [ ] `rank.py` reproduces the CSV in a clean Docker container within 5 min / 16 GB / no network.
- [ ] Reasoning passes the automated grounding check (no hallucinated skills/employers/numbers) and varies row to row.
- [ ] Sandbox link runs a ≤100-candidate sample end-to-end.
- [ ] `submission_metadata.yaml` + portal metadata complete and honest (AI tools declared).
- [ ] README has the single repro command; git history shows genuine iteration.
- [ ] Held-out NDCG@10/@50 on our silver set is meaningfully above the BM25 baseline.
