"""Turn a candidate record into the text used for embedding and BM25.

A single function so the dense index and the sparse index see *exactly* the same
text, and so retrieval text is reproducible between the offline build and any
later inspection.
"""
from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def candidate_to_text(c: dict[str, Any]) -> str:
    """Concatenate the human-meaningful fields: headline, summary, titles,
    company names, role descriptions, and skill names."""
    profile = c.get("profile", {}) or {}
    parts: list[str] = []

    headline = profile.get("headline")
    if headline:
        parts.append(str(headline))

    summary = profile.get("summary")
    if summary:
        parts.append(str(summary))

    current_title = profile.get("current_title")
    if current_title:
        parts.append(str(current_title))

    for role in c.get("career_history", []) or []:
        seg = " ".join(
            str(role.get(k, ""))
            for k in ("title", "company", "industry", "description")
            if role.get(k)
        )
        if seg.strip():
            parts.append(seg)

    skills = [str(s.get("name", "")) for s in c.get("skills", []) or [] if s.get("name")]
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    for edu in c.get("education", []) or []:
        seg = " ".join(
            str(edu.get(k, "")) for k in ("degree", "field_of_study", "institution") if edu.get(k)
        )
        if seg.strip():
            parts.append(seg)

    return "\n".join(parts).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25. Keeps tech tokens like c++, .net, e5."""
    return _WORD_RE.findall(text.lower())
