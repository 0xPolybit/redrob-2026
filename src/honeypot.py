"""General impossibility / trap detection.

The dataset seeds ~80 honeypots with *logically impossible* profiles that are
forced to relevance tier 0 in the ground truth; >10% of them in the top 100
disqualifies the submission. We deliberately do **not** special-case the 80 —
we implement general consistency checks, exactly as the spec recommends.

`honeypot_flags(candidate)` returns (is_honeypot, reasons). A True result means
the profile contains an internal contradiction no real career could.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

# A little slack so ordinary rounding / overlapping jobs don't trip the checks.
_EXPERIENCE_SLACK_MONTHS = 18      # career months may exceed stated YoE by this much
_FUTURE_SLACK_DAYS = 31
_TODAY = date(2026, 6, 27)         # dataset reference "today"
_MAX_PLAUSIBLE_TENURE_MONTHS = 50 * 12


def _parse_date(s: Any) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def honeypot_flags(c: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (is_honeypot, reasons). True ⇒ force to the bottom of the ranking."""
    reasons: list[str] = []
    profile = c.get("profile", {}) or {}
    yoe = profile.get("years_of_experience")
    history = c.get("career_history", []) or []

    total_role_months = 0
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        dur = role.get("duration_months")

        # 1. impossible date range: end strictly before start
        if start and end and end < start:
            reasons.append(f"role end_date {end} precedes start_date {start}")

        # 2. start date in the future
        if start and start > date(_TODAY.year, _TODAY.month, _TODAY.day):
            reasons.append(f"role start_date {start} is in the future")

        # 3. single tenure longer than any human career
        if isinstance(dur, (int, float)) and dur > _MAX_PLAUSIBLE_TENURE_MONTHS:
            reasons.append(f"role duration_months {dur} exceeds a 50-year career")

        # 4. stated duration grossly inconsistent with its own date span
        if start and dur is not None:
            span_end = end or _TODAY
            span_months = (span_end.year - start.year) * 12 + (span_end.month - start.month)
            if span_months >= 0 and dur - span_months > _EXPERIENCE_SLACK_MONTHS:
                reasons.append(
                    f"role duration_months {dur} far exceeds its date span (~{span_months}m)"
                )

        if isinstance(dur, (int, float)):
            total_role_months += dur

    # 5. summed tenure can't be reconciled with stated years_of_experience.
    #    People hold overlapping/part-time roles, so we only flag the *excess*
    #    beyond stated experience (you can't have worked more months than you've lived a career).
    if isinstance(yoe, (int, float)) and yoe >= 0:
        if total_role_months - yoe * 12 > _EXPERIENCE_SLACK_MONTHS:
            reasons.append(
                f"career history sums to {total_role_months}m but years_of_experience is {yoe}"
            )

    # 6. "expert" in a skill with 0 months of use — and several of them.
    expert_zero = 0
    for s in c.get("skills", []) or []:
        if str(s.get("proficiency", "")).lower() == "expert" and s.get("duration_months", 1) == 0:
            expert_zero += 1
    if expert_zero >= 3:
        reasons.append(f"{expert_zero} skills are 'expert' with 0 months of use")

    # 7. claims experience but has no career history at all
    if isinstance(yoe, (int, float)) and yoe >= 3 and not history:
        reasons.append(f"years_of_experience {yoe} but empty career_history")

    # 8. last active before they even signed up
    sig = _parse_date((c.get("redrob_signals", {}) or {}).get("signup_date"))
    last = _parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
    if sig and last and last < sig:
        reasons.append(f"last_active_date {last} precedes signup_date {sig}")

    return (len(reasons) > 0, reasons)
