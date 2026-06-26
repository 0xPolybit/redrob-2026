"""OFFLINE: grounded reasoning strings for the final top 100.

The reasoning column is graded at Stage 4 for specific facts, JD connection,
honest concerns, no hallucination, variation, and tone-matching-rank. We feed
Claude ONLY the candidate's real fields + their feature values + assigned rank,
then run an automated grounding check: any number named in the reasoning must
trace back to the record. Failures fall back to a deterministic, fact-only
template (zero hallucination risk).

Reads artifacts/top100.json (written by rank.py) + candidates.jsonl.
Writes artifacts/reasoning.parquet (candidate_id, reasoning). Re-run rank.py
afterwards to attach them.
"""
from __future__ import annotations

import json
import re

from . import config, features
from .llm import get_client, structured_call

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"reasoning": {"type": "string"}},
    "required": ["reasoning"],
}

_SYSTEM = (
    "You write a 1-2 sentence hiring justification for a candidate's rank against "
    "a Senior AI Engineer JD. Rules: cite SPECIFIC facts from the candidate "
    "(years, current title, named skills/employers, signal values); connect to a "
    "JD requirement; acknowledge a real concern when one exists; never mention a "
    "skill/employer/number that is not in the provided record. Match tone to "
    "rank — confident near rank 1, hedged near rank 100. Be plain and honest, "
    "not impressive."
)

_USER = (
    "JD must-haves: {must}\n\nRank: {rank} of 100\n\nCANDIDATE RECORD:\n{cand}\n\n"
    "Feature highlights: years={yoe}, applied_ml_years={ml}, "
    "core_ai_skills={core}, product_ratio={prod}, keyword_stuffer_score={stuff}, "
    "recruiter_response_rate={rr}, last_active_days={la}.\n\n"
    "Write the reasoning."
)


def _record_blob(c: dict) -> str:
    p = c.get("profile", {}) or {}
    parts = [str(p.get(k, "")) for k in ("anonymized_name", "headline", "summary",
                                          "current_title", "current_company",
                                          "current_industry", "location", "country",
                                          "years_of_experience")]
    for r in c.get("career_history", []) or []:
        parts += [str(r.get(k, "")) for k in ("title", "company", "industry",
                                              "duration_months", "description")]
    for s in c.get("skills", []) or []:
        parts.append(str(s.get("name", "")))
    sig = c.get("redrob_signals", {}) or {}
    parts += [str(v) for v in sig.values() if not isinstance(v, dict)]
    parts += [f"{v:.2f}" for v in sig.values() if isinstance(v, (int, float))]
    return " ".join(parts).lower()


def _allowed_numbers(c: dict) -> list[float]:
    nums: list[float] = []
    p = c.get("profile", {}) or {}
    if isinstance(p.get("years_of_experience"), (int, float)):
        nums.append(float(p["years_of_experience"]))
    for r in c.get("career_history", []) or []:
        if isinstance(r.get("duration_months"), (int, float)):
            nums.append(float(r["duration_months"]))
            nums.append(round(r["duration_months"] / 12.0, 1))
    sig = c.get("redrob_signals", {}) or {}
    for v in sig.values():
        if isinstance(v, (int, float)):
            nums.append(float(v))
    return nums


def _is_grounded(reasoning: str, c: dict) -> bool:
    blob = _record_blob(c)
    allowed = _allowed_numbers(c)
    for tok in _NUM_RE.findall(reasoning):
        val = float(tok)
        if tok in blob:
            continue
        if any(abs(val - a) <= 0.6 for a in allowed):
            continue
        return False
    return True


def _template(c: dict, feat: dict) -> str:
    p = c.get("profile", {}) or {}
    skills = [s.get("name") for s in (c.get("skills") or [])
              if str(s.get("name", "")).lower() in config.CORE_AI_SKILLS][:3]
    skill_str = ", ".join(skills) if skills else "general ML"
    concern = ""
    if feat.get("keyword_stuffer_score", 0) >= 0.5:
        concern = " Concern: skills list outpaces demonstrated evidence."
    elif feat.get("product_ratio", 1) < 0.4:
        concern = " Concern: largely services/consulting background."
    elif feat.get("last_active_days", 0) > 180:
        concern = " Concern: low recent platform activity."
    return (
        f"{p.get('current_title','Candidate')} with "
        f"{p.get('years_of_experience','?')} yrs; matched on {skill_str}; "
        f"recruiter response {feat.get('recruiter_response_rate',0):.2f}.{concern}"
    ).strip()


def main() -> None:
    top_path = config.ARTIFACTS_DIR / "top100.json"
    if not top_path.exists():
        raise SystemExit("artifacts/top100.json missing — run rank.py once first.")
    top_ids = json.loads(top_path.read_text(encoding="utf-8"))
    rank_of = {cid: i + 1 for i, cid in enumerate(top_ids)}
    top_set = set(top_ids)

    cands: dict[str, dict] = {}
    with open(config.resolve_candidates_path(), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in top_set:
                cands[c["candidate_id"]] = c

    rubric = json.loads(config.JD_RUBRIC.read_text(encoding="utf-8"))
    must = "; ".join(rubric.get("must_haves", []))
    client = get_client()

    out = []
    for cid in top_ids:
        c = cands.get(cid)
        if c is None:
            continue
        feat = features.extract_features(c)
        user = _USER.format(
            must=must, rank=rank_of[cid], cand=json.dumps(c, ensure_ascii=False)[:6000],
            yoe=feat["years_of_experience"], ml=feat["est_applied_ml_years"],
            core=int(feat["num_core_ai_skills"]), prod=round(feat["product_ratio"], 2),
            stuff=round(feat["keyword_stuffer_score"], 2),
            rr=round(feat["recruiter_response_rate"], 2), la=int(feat["last_active_days"]),
        )
        reasoning = None
        for _ in range(2):  # one regeneration attempt on grounding failure
            try:
                res = structured_call(client, system=_SYSTEM, user=user,
                                      schema=_SCHEMA, max_tokens=1200)
                candidate_text = res["reasoning"].strip()
            except Exception as e:  # noqa: BLE001 — fall back rather than abort the batch
                print(f"  {cid}: LLM error ({e}); using template")
                break
            if _is_grounded(candidate_text, c):
                reasoning = candidate_text
                break
        if reasoning is None:
            reasoning = _template(c, feat)
        out.append({"candidate_id": cid, "reasoning": reasoning})
        print(f"  rank {rank_of[cid]:>3} {cid}: {reasoning[:80]}")

    import polars as pl

    pl.DataFrame(out).write_parquet(config.REASONING_PARQUET)
    print(f"Wrote {config.REASONING_PARQUET} ({len(out)} rows)")


if __name__ == "__main__":
    main()
