"""Seeded impossible profiles must be caught; a plausible profile must not be."""
from src.honeypot import honeypot_flags


def _base():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"years_of_experience": 6.0, "current_title": "ML Engineer"},
        "career_history": [
            {"company": "ProductCo", "title": "ML Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "Software", "company_size": "201-500", "description": "Built search."}
        ],
        "skills": [{"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 36}],
        "redrob_signals": {"signup_date": "2019-01-01", "last_active_date": "2026-06-01"},
    }


def test_plausible_profile_not_flagged():
    is_hp, reasons = honeypot_flags(_base())
    assert not is_hp, reasons


def test_tenure_exceeds_experience():
    c = _base()
    c["profile"]["years_of_experience"] = 3.0
    c["career_history"][0]["duration_months"] = 200  # 16y of tenure, 3y stated
    is_hp, reasons = honeypot_flags(c)
    assert is_hp and any("years_of_experience" in r for r in reasons)


def test_expert_with_zero_months():
    c = _base()
    c["skills"] = [
        {"name": s, "proficiency": "expert", "endorsements": 1, "duration_months": 0}
        for s in ("RAG", "FAISS", "Pinecone", "BM25")
    ]
    is_hp, reasons = honeypot_flags(c)
    assert is_hp and any("expert" in r for r in reasons)


def test_end_before_start():
    c = _base()
    c["career_history"][0]["start_date"] = "2022-01-01"
    c["career_history"][0]["end_date"] = "2020-01-01"
    is_hp, reasons = honeypot_flags(c)
    assert is_hp and any("precedes start_date" in r for r in reasons)


def test_last_active_before_signup():
    c = _base()
    c["redrob_signals"]["signup_date"] = "2025-01-01"
    c["redrob_signals"]["last_active_date"] = "2020-01-01"
    is_hp, reasons = honeypot_flags(c)
    assert is_hp and any("signup_date" in r for r in reasons)
