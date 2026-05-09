"""MergeOrInsert: dedupe + confidence aggregation (Algorithm 3 line 19)."""

from __future__ import annotations

from typing import Iterable

from ..inducer.schema import KnowledgeEntry
from .validation import _payload_signature


def find_duplicate(
    candidate: KnowledgeEntry,
    store: Iterable[KnowledgeEntry],
) -> KnowledgeEntry | None:
    """Find an existing entry with same type + same failure_type + identical
    payload signature. If found, the candidate should be merged into it
    instead of inserted as a new entry.
    """

    sig = _payload_signature(candidate)
    for other in store:
        if other.type != candidate.type:
            continue
        if other.failure_type != candidate.failure_type:
            continue
        if _payload_signature(other) != sig:
            continue
        return other
    return None


def merge_into(existing: KnowledgeEntry, candidate: KnowledgeEntry) -> KnowledgeEntry:
    """Combine confidences and supporting feedback in-place on ``existing``."""

    existing.support_count += 1
    # Weighted moving average of confidence
    existing.rho = float(
        (existing.rho * (existing.support_count - 1) + candidate.rho)
        / max(1, existing.support_count)
    )
    # Merge supporting feedback ids (deduped)
    seen = set(existing.E)
    for fid in candidate.E:
        if fid not in seen:
            existing.E.append(fid)
            seen.add(fid)
    # Merge trigger contexts
    seen_ctx = {str(x).lower() for x in existing.c}
    for ctx in candidate.c:
        if str(ctx).lower() not in seen_ctx:
            existing.c.append(ctx)
            seen_ctx.add(str(ctx).lower())
    existing.updated_at = max(existing.updated_at, candidate.created_at)
    return existing
