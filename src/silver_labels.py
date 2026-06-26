"""OFFLINE: generate 'silver' relevance labels for a sampled subset.

Ground truth is hidden, so we sample ~2,000 candidates spanning the rubric-score
range (and deliberately include honeypots + likely keyword-stuffers) and have
Claude grade each on the same 0-5 relevance-tier scale the organizers use. These
labels train the LightGBM ranker. Allowed: offline development, not the timed step.

Grading runs through the Batches API (50% cheaper, one poll loop) by default.
Writes artifacts/silver_labels.parquet (candidate_id, tier, reason).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import config, features
from .llm import get_client

_TIER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tier": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["tier", "reason"],
}

_SYSTEM = (
    "You grade candidates against a Senior AI Engineer rubric on a 0-5 relevance "
    "scale. The JD is adversarial: 'most AI keywords' is a TRAP.\n"
    "Tier 5 = ideal: 6-8y, 4-5y applied ML at PRODUCT companies, shipped an "
    "end-to-end ranking/search/recommendation system to real users, strong "
    "production retrieval + vector-DB + eval literacy. Reward strong candidates "
    "even when they never write 'RAG'/'Pinecone' but clearly built such systems.\n"
    "Tier 3-4 = relevant with gaps. Tier 1-2 = weak/adjacent.\n"
    "Tier 0 = irrelevant, a keyword-stuffer (AI skills listed but career/title "
    "don't back them up), a hard disqualifier (pure research, consulting-only, "
    "recent-LangChain-only, CV/speech/robotics primary), or a logically "
    "impossible profile (honeypot). Judge the gap between what the profile SAYS "
    "and what it MEANS."
)

_USER_TEMPLATE = (
    "RUBRIC:\n{rubric}\n\nCANDIDATE:\n{cand}\n\n"
    "Assign the single best relevance tier (0-5) and a one-sentence reason."
)


def _grading_blob(c: dict) -> str:
    """Compact, token-bounded candidate summary for the grader."""
    p = c.get("profile", {}) or {}
    s = c.get("redrob_signals", {}) or {}
    lines = [
        f"name: {p.get('anonymized_name')}",
        f"headline: {p.get('headline')}",
        f"years_of_experience: {p.get('years_of_experience')}",
        f"current: {p.get('current_title')} @ {p.get('current_company')} "
        f"({p.get('current_industry')}, {p.get('current_company_size')})",
        f"location: {p.get('location')}, {p.get('country')}",
        f"summary: {str(p.get('summary',''))[:600]}",
        "career_history:",
    ]
    for r in (c.get("career_history") or [])[:6]:
        lines.append(
            f"  - {r.get('title')} @ {r.get('company')} ({r.get('industry')}, "
            f"{r.get('duration_months')}m): {str(r.get('description',''))[:300]}"
        )
    skills = ", ".join(
        f"{sk.get('name')}[{sk.get('proficiency')}]" for sk in (c.get("skills") or [])[:25]
    )
    lines.append(f"skills: {skills}")
    assess = s.get("skill_assessment_scores", {}) or {}
    lines.append(f"skill_assessment_scores: {json.dumps(assess)[:400]}")
    lines.append(
        "signals: "
        f"last_active={s.get('last_active_date')}, "
        f"recruiter_response_rate={s.get('recruiter_response_rate')}, "
        f"open_to_work={s.get('open_to_work_flag')}, "
        f"notice_days={s.get('notice_period_days')}, "
        f"github={s.get('github_activity_score')}"
    )
    return "\n".join(lines)


def _select_sample(n: int, seed: int = 13) -> set[str]:
    """Stratify by rubric score; force-include honeypots + likely stuffers."""
    import polars as pl

    df = pl.read_parquet(config.FEATURES_PARQUET)
    rows = df.to_dicts()
    rng = np.random.default_rng(seed)

    scores = np.array([features.rubric_score(r) for r in rows])
    ids = np.array([r["candidate_id"] for r in rows])

    chosen: set[str] = set()
    # force-include traps so the model learns to reject them
    for r in rows:
        if r.get("honeypot", 0) >= 1 or r.get("keyword_stuffer_score", 0) >= 0.5:
            chosen.add(r["candidate_id"])
    forced = len(chosen)

    # stratified across 10 score quantiles for the remainder
    remaining = max(0, n - forced)
    if remaining:
        bins = np.quantile(scores, np.linspace(0, 1, 11))
        bin_idx = np.clip(np.digitize(scores, bins[1:-1]), 0, 9)
        per_bin = remaining // 10 + 1
        for b in range(10):
            pool = ids[bin_idx == b]
            pool = [i for i in pool if i not in chosen]
            if pool:
                take = rng.choice(pool, size=min(per_bin, len(pool)), replace=False)
                chosen.update(take.tolist())
    print(f"Sample: {len(chosen)} (forced traps/honeypots: {forced})")
    return set(list(chosen)[:max(n, forced)])


def _load_sample_candidates(sample_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(config.resolve_candidates_path(), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in sample_ids:
                out[c["candidate_id"]] = c
    return out


def _grade_batches(client, rubric: dict, cands: dict[str, dict]) -> dict[str, dict]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    rubric_str = json.dumps(rubric, indent=2)
    requests = [
        Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=config.CLAUDE_MODEL,
                max_tokens=1500,
                thinking={"type": "adaptive"},
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": _USER_TEMPLATE.format(rubric=rubric_str, cand=_grading_blob(c)),
                }],
                output_config={"format": {"type": "json_schema", "schema": _TIER_SCHEMA}},
            ),
        )
        for cid, c in cands.items()
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch {batch.id} created with {len(requests)} requests; polling ...")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  status={b.processing_status} processing={b.request_counts.processing}")
        time.sleep(30)

    labels: dict[str, dict] = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((blk.text for blk in msg.content if blk.type == "text"), None)
        if text:
            try:
                parsed = json.loads(text)
                labels[result.custom_id] = parsed
            except json.JSONDecodeError:
                pass
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.SILVER_SAMPLE_SIZE)
    args = ap.parse_args()

    rubric = json.loads(config.JD_RUBRIC.read_text(encoding="utf-8"))
    sample_ids = _select_sample(args.n)
    print(f"Loading {len(sample_ids)} sampled candidates from JSONL ...")
    cands = _load_sample_candidates(sample_ids)

    client = get_client()
    labels = _grade_batches(client, rubric, cands)
    print(f"Graded {len(labels)} candidates.")

    import polars as pl

    df = pl.DataFrame(
        [{"candidate_id": cid, "tier": int(v["tier"]), "reason": v.get("reason", "")}
         for cid, v in labels.items()]
    )
    df.write_parquet(config.SILVER_LABELS)
    print(f"Wrote {config.SILVER_LABELS} {df.shape}")
    print("Tier distribution:", df["tier"].value_counts().sort("tier").to_dicts())


if __name__ == "__main__":
    main()
