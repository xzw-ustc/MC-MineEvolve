"""Robust JSON parsing helpers for LLM outputs.

LLMs sometimes wrap JSON in ```json fences``` or include a leading prose
sentence. ``parse_json`` tries the most permissive recovery short of
hallucinating fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json(text: str) -> Any | None:
    """Try several recovery strategies; return None if nothing works."""

    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        pass

    fence = _FENCE_RE.search(s)
    if fence is not None:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass

    start = s.find("[")
    end = s.rfind("]")
    if 0 <= start < end:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass

    return None


def coerce_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = parse_json(value)
        if isinstance(parsed, dict):
            return parsed
    return None
