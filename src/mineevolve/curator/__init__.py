"""Curator module (paper Section 3.3 + Algorithm 3 stages 1-2).

The ``Curator`` orchestrates validate -> merge-or-insert -> retrieve. It
exposes a small surface used by the server agent (``ingest`` and
``retrieve``) plus persistence helpers.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from ..inducer.schema import KnowledgeEntry
from .merge import find_duplicate, merge_into
from .retrieval import (
    RetrievalContext,
    retrieval_context_from_state,
    retrieve,
    split_skills_remedies,
)
from .store import KnowledgeStore
from .validation import accept


class Curator:
    """Validate / merge / retrieve external behavioural knowledge."""

    def __init__(
        self,
        store: KnowledgeStore,
        executor_actions: Iterable[str] = (),
    ) -> None:
        self.store = store
        self.executor_actions = tuple(executor_actions)

    # ------------------------------------------------------------------
    # Ingestion (Algorithm 3 stage 1)
    # ------------------------------------------------------------------

    def ingest(
        self,
        candidates: Sequence[KnowledgeEntry],
        state_fields: Mapping[str, object] | None = None,
    ) -> dict:
        """Validate and (merge or insert) candidate entries.

        Returns a small report dict { accepted: [...], rejected: [...] }.
        """

        accepted: List[str] = []
        rejected: List[dict] = []
        existing = self.store.all()

        for k in candidates:
            ok, reason = accept(
                k=k,
                store=existing,
                state_fields=state_fields or {},
                executor_actions=self.executor_actions,
            )
            if not ok:
                rejected.append({"id": k.knowledge_id, "reason": reason})
                continue

            dup = find_duplicate(k, existing)
            if dup is not None:
                merged = merge_into(dup, k)
                self.store.insert(merged)  # overwrite same-id entry
                accepted.append(merged.knowledge_id)
            else:
                self.store.insert(k)
                accepted.append(k.knowledge_id)
                existing.append(k)

        self.store.flush()
        return {"accepted": accepted, "rejected": rejected}

    # ------------------------------------------------------------------
    # Retrieval (Algorithm 3 stage 2)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        task_goal: str,
        current_subgoal: str,
        inventory: Mapping[str, int] | None,
        recent_failure_types: Sequence[str] = (),
        budget_tokens: int = 512,
        top_k: int = 16,
    ) -> dict:
        ctx = retrieval_context_from_state(
            task_goal=task_goal,
            current_subgoal=current_subgoal,
            inventory=inventory,
            recent_failure_types=recent_failure_types,
        )
        entries = retrieve(self.store.all(), ctx, budget_tokens=budget_tokens, top_k=top_k)
        skills, remedies = split_skills_remedies(entries)
        self.store.update_usage(k.knowledge_id for k in entries)
        return {
            "skills": [k.to_dict() for k in skills],
            "remedies": [k.to_dict() for k in remedies],
            "skill_ids": [k.knowledge_id for k in skills],
            "remedy_ids": [k.knowledge_id for k in remedies],
            "stats": self.store.stats(),
        }


__all__ = [
    "Curator",
    "KnowledgeStore",
    "RetrievalContext",
]
