"""Deterministic, vectorizable feature extraction.

Every feature is cheap and computed from the candidate record alone (no LLM,
no network). Shared by the offline build (writes one row per candidate into
features.parquet) and conceptually by rank.py (which just reads those rows).

Grouped into: career structure, trap/quality, disqualifier flags, and
behavioral signals. Also exposes a transparent hand-weighted `rubric_score`
(the insurance-policy scorer and the blend partner for the LightGBM model) and
an `availability_multiplier` derived from the Redrob engagement signals.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from . import config
from .build_text import tokenize

_TODAY = date(2026, 6, 27)

# Title vocabularies for coherence / hands-on detection.
_HANDS_ON_TITLE = {
    "engineer", "developer", "scientist", "programmer", "sde", "mle",
    "ml engineer", "data scientist", "research engineer", "applied scientist",
}
_DRIFT_TITLE = {
    "architect", "lead", "manager", "director", "head", "vp", "principal",
    "cto", "chief",
}
_NON_TECH_TITLE = {
    "marketing", "sales", "hr", "human resources", "recruiter", "content",
    "writer", "designer", "accountant", "finance", "operations", "admin",
    "business development", "customer success", "product manager",
}


def _parse_date(s: Any) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(d: date | None) -> float:
    if d is None:
        return 365.0 * 5  # treat unknown as very stale
    return float((_TODAY - d).days)


def _contains_any(text: str, vocab) -> bool:
    return any(term in text for term in vocab)


def _count_hits(text: str, vocab) -> int:
    return sum(1 for term in vocab if term in text)


def extract_features(c: dict[str, Any]) -> dict[str, float]:
    """Return a flat dict of non-semantic numeric features for one candidate."""
    profile = c.get("profile", {}) or {}
    signals = c.get("redrob_signals", {}) or {}
    history = c.get("career_history", []) or []
    skills = c.get("skills", []) or []

    f: dict[str, float] = {}

    # ---- career structure ----
    yoe = float(profile.get("years_of_experience") or 0.0)
    f["years_of_experience"] = yoe
    f["num_roles"] = float(len(history))

    desc_text = " ".join(str(r.get("description", "")).lower() for r in history)
    title_text = " ".join(str(r.get("title", "")).lower() for r in history)
    current_title = str(profile.get("current_title", "")).lower()

    # applied-ML months: tenure in roles whose title/description show ML/AI work.
    ml_months = 0.0
    services_months = 0.0
    product_months = 0.0
    total_months = 0.0
    for r in history:
        dur = float(r.get("duration_months") or 0)
        total_months += dur
        blob = (str(r.get("title", "")) + " " + str(r.get("description", ""))).lower()
        if _contains_any(blob, config.AI_KEYWORDS):
            ml_months += dur
        industry = str(r.get("industry", "")).lower()
        company = str(r.get("company", "")).lower()
        is_services = industry in config.SERVICES_INDUSTRIES or _contains_any(
            company, config.CONSULTING_FIRMS
        )
        if is_services:
            services_months += dur
        else:
            product_months += dur

    f["est_applied_ml_years"] = round(ml_months / 12.0, 3)
    f["product_ratio"] = round(product_months / total_months, 4) if total_months else 0.0
    f["services_ratio"] = round(services_months / total_months, 4) if total_months else 0.0

    cur_industry = str(profile.get("current_industry", "")).lower()
    f["current_is_services"] = 1.0 if cur_industry in config.SERVICES_INDUSTRIES else 0.0

    f["title_is_hands_on"] = (
        1.0 if _contains_any(current_title, _HANDS_ON_TITLE) else 0.0
    )
    f["title_is_drift"] = 1.0 if _contains_any(current_title, _DRIFT_TITLE) else 0.0

    # ---- trap / quality ----
    skill_names = [str(s.get("name", "")).lower() for s in skills]
    skill_blob = " ".join(skill_names)
    f["num_skills_total"] = float(len(skills))
    f["num_core_ai_skills"] = float(sum(1 for s in skill_names if s in config.CORE_AI_SKILLS))
    f["num_ai_keywords"] = float(_count_hits(skill_blob, config.AI_KEYWORDS))

    assessments = signals.get("skill_assessment_scores", {}) or {}
    assess_vals = [float(v) for v in assessments.values() if isinstance(v, (int, float))]
    f["skill_assessment_avg"] = round(sum(assess_vals) / len(assess_vals), 2) if assess_vals else -1.0
    core_assess = [
        float(v)
        for k, v in assessments.items()
        if str(k).lower() in config.AI_KEYWORDS and isinstance(v, (int, float))
    ]
    f["skill_assessment_core_avg"] = round(sum(core_assess) / len(core_assess), 2) if core_assess else -1.0

    # Keyword-stuffer: many AI skills listed, little supporting evidence in the
    # career descriptions and weak/absent assessment scores.
    ai_evidence = _count_hits(desc_text, config.AI_KEYWORDS)
    f["ai_evidence_in_descriptions"] = float(ai_evidence)
    listed = f["num_ai_keywords"]
    assess_support = f["skill_assessment_core_avg"] / 100.0 if f["skill_assessment_core_avg"] >= 0 else 0.0
    f["keyword_stuffer_score"] = round(
        max(0.0, listed - ai_evidence - assess_support * listed) / (listed + 1.0), 4
    )

    # Title–skill coherence: a non-technical current title plus many AI skills is incoherent.
    non_tech = _contains_any(current_title, _NON_TECH_TITLE)
    if non_tech and listed >= 4:
        coherence = 0.0
    elif f["title_is_hands_on"] and f["num_core_ai_skills"] >= 1:
        coherence = 1.0
    elif non_tech:
        coherence = 0.3
    else:
        coherence = 0.6
    f["title_skill_coherence"] = coherence

    # ---- disqualifier flags (also used by the rule layer) ----
    summary = str(profile.get("summary", "")).lower()
    full_blob = " ".join([summary, desc_text, title_text, skill_blob])

    research_hits = _count_hits(full_blob, config.RESEARCH_TERMS)
    f["disq_research_only"] = 1.0 if (research_hits >= 2 and product_months < 12) else 0.0
    f["disq_consulting_only"] = 1.0 if (services_months > 0 and product_months == 0 and total_months > 0) else 0.0

    cv_hits = _count_hits(full_blob, config.CV_SPEECH_ROBOTICS)
    nlp_ir_hits = _count_hits(full_blob, config.CORE_AI_SKILLS)
    f["disq_cv_speech_robotics_primary"] = 1.0 if (cv_hits >= 2 and nlp_ir_hits == 0) else 0.0

    langchain_hits = _count_hits(full_blob, config.LANGCHAIN_TERMS)
    pre_llm_evidence = ml_months >= 24
    f["disq_recent_langchain_only"] = 1.0 if (langchain_hits >= 1 and not pre_llm_evidence and yoe < 6) else 0.0

    f["disq_architect_drift"] = 1.0 if (f["title_is_drift"] and not f["title_is_hands_on"]) else 0.0

    # ---- education ----
    edu = c.get("education", []) or []
    f["tier1_education"] = 1.0 if any(str(e.get("tier", "")).lower() == "tier_1" for e in edu) else 0.0

    # ---- behavioral signals ----
    last_active_days = _days_since(_parse_date(signals.get("last_active_date")))
    f["last_active_days"] = last_active_days
    f["recency_decay"] = round(math.exp(-last_active_days / 180.0), 4)
    f["recruiter_response_rate"] = float(signals.get("recruiter_response_rate") or 0.0)
    f["interview_completion_rate"] = float(signals.get("interview_completion_rate") or 0.0)
    oar = signals.get("offer_acceptance_rate")
    f["offer_acceptance_rate"] = float(oar) if isinstance(oar, (int, float)) and oar >= 0 else -1.0
    f["open_to_work"] = 1.0 if signals.get("open_to_work_flag") else 0.0
    f["profile_completeness"] = float(signals.get("profile_completeness_score") or 0.0)
    gh = signals.get("github_activity_score")
    f["github_activity"] = float(gh) if isinstance(gh, (int, float)) and gh >= 0 else -1.0
    f["saved_by_recruiters_30d"] = float(signals.get("saved_by_recruiters_30d") or 0.0)
    f["search_appearance_30d"] = float(signals.get("search_appearance_30d") or 0.0)
    f["profile_views_30d"] = float(signals.get("profile_views_received_30d") or 0.0)
    notice = float(signals.get("notice_period_days") or 0.0)
    f["notice_period_days"] = notice
    f["notice_ok"] = 1.0 if notice <= 30 else 0.0
    f["willing_to_relocate"] = 1.0 if signals.get("willing_to_relocate") else 0.0

    location = str(profile.get("location", "")).lower()
    f["location_preferred"] = 1.0 if _contains_any(location, config.PREFERRED_LOCATIONS) else 0.0

    return f


# Columns that are NOT model inputs (used only for the rule layer / joins).
NON_MODEL_COLUMNS = {"candidate_id", "honeypot", "availability_mult"}


def availability_multiplier(f: dict[str, float]) -> float:
    """Scale a candidate's score by how reachable/available they actually are.

    A perfect-on-paper but unreachable candidate (stale login, low response
    rate) is, for hiring, not available — down-weight, don't drop.
    Returns a multiplier in roughly [0.4, 1.1].
    """
    recency = f.get("recency_decay", 0.5)                 # 0..1
    response = f.get("recruiter_response_rate", 0.0)      # 0..1
    completeness = f.get("profile_completeness", 50.0) / 100.0
    open_flag = f.get("open_to_work", 0.0)

    base = 0.5 + 0.25 * recency + 0.15 * response + 0.10 * completeness
    if open_flag:
        base += 0.05
    return float(max(0.4, min(1.1, base)))


def rubric_score(f: dict[str, float]) -> float:
    """Transparent hand-weighted relevance score in ~[0, 1].

    Used as the insurance-policy baseline and as the blend partner for the
    LightGBM model. Reads semantic-similarity keys (sim_*) when present but
    works without them. Mirrors the JD: reward production retrieval/ranking +
    product-company applied ML; punish keyword-stuffers and disqualifiers.
    """
    score = 0.0

    # semantic fit (present at rank time; absent in a pure-feature smoke test)
    score += 0.30 * f.get("sim_pooled", 0.0)
    score += 0.15 * f.get("sim_req_max", 0.0)

    # core production-AI signal
    score += 0.10 * min(f.get("num_core_ai_skills", 0.0) / 4.0, 1.0)
    score += 0.10 * min(f.get("est_applied_ml_years", 0.0) / 5.0, 1.0)
    score += 0.08 * f.get("product_ratio", 0.0)
    score += 0.05 * f.get("title_skill_coherence", 0.0)
    score += 0.04 * (1.0 if 5 <= f.get("years_of_experience", 0.0) <= 9 else 0.0)
    if f.get("skill_assessment_core_avg", -1.0) >= 0:
        score += 0.05 * (f["skill_assessment_core_avg"] / 100.0)

    # engagement / reachability
    score += 0.04 * f.get("recency_decay", 0.0)
    score += 0.03 * f.get("recruiter_response_rate", 0.0)
    score += 0.02 * f.get("location_preferred", 0.0)

    # penalties (the traps)
    score -= 0.20 * f.get("keyword_stuffer_score", 0.0)
    score -= 0.15 * f.get("disq_research_only", 0.0)
    score -= 0.15 * f.get("disq_consulting_only", 0.0)
    score -= 0.12 * f.get("disq_cv_speech_robotics_primary", 0.0)
    score -= 0.10 * f.get("disq_recent_langchain_only", 0.0)
    score -= 0.08 * f.get("disq_architect_drift", 0.0)

    return float(max(0.0, min(1.0, score)))
