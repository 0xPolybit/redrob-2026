"""Thin Anthropic-SDK helpers used only by the OFFLINE scripts.

Never imported by rank.py — the timed step makes no LLM/network calls.
"""
from __future__ import annotations

import html
import json
import os
import re
import zipfile
from typing import Any

from . import config


def docx_to_text(path: str | os.PathLike) -> str:
    """Extract plain text from a .docx without any third-party dependency."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    xml = html.unescape(xml)
    return re.sub(r"\n{3,}", "\n\n", xml).strip()


def get_client():
    """Return an Anthropic client, raising a clear error if the key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. The offline LLM steps "
            "(jd_parse / silver_labels / reasoning) require it. "
            "Set it and re-run; rank.py itself never needs it."
        )
    import anthropic  # imported lazily so rank.py never pulls it in

    return anthropic.Anthropic()


def structured_call(
    client,
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """One structured-output Messages call returning parsed JSON.

    Uses adaptive thinking and output_config.format json_schema so the response
    is guaranteed to validate against `schema`.
    """
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)
