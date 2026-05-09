"""Remedy induction: failure / stagnation / deadlock feedback -> KnowledgeEntry(remedy).

Implements Algorithm 2 stage 3 (lines 22-28) for the remedy branch and
the trigger condition of paper Eq. (5):

    (1/h) * sum_{j=i-h+1}^{i} I[y_j = 0] >= eta_fail   OR   l_i = 1

Plus the task-level deadlock trigger from paper Figure 3 (the "Deadlock
Pattern" entry under Task-Level remedy induction).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..monitor.feedback import TypedFeedback
from .prompts import (
    REMEDY_SYSTEM_PROMPT,
    render_remedy_user_prompt,
)
from .schema import KnowledgeEntry, RemedyCategory, RemedyScope


def need_remedy(
    segment: Sequence[TypedFeedback],
    eta_fail: float = 0.5,
    deadlock_signal: Mapping | None = None,
) -> bool:
    """Eq. (5) OR cross-subgoal deadlock.

    Returns True if any of:
      * latest item is stagnant (l_i == 1);
      * recent failure rate >= eta_fail;
      * a deadlock signal is provided (cross-subgoal repeating failure).
    """

    if deadlock_signal:
        return True
    if not segment:
        return False
    if segment[-1].is_stagnant():
        return True
    failures = sum(1 for e in segment if not e.is_successful())
    return float(failures) / float(len(segment)) >= float(eta_fail)


def _normalize_scope(value: object) -> str:
    s = str(value or "").strip().lower()
    if s in RemedyScope.ALL:
        return s
    return RemedyScope.UNFINISHED_PLAN_SUFFIX


def _normalize_category(value: object) -> str:
    c = str(value or "").strip().lower()
    if c in RemedyCategory.ALL:
        return c
    return RemedyCategory.GENERIC


def build_remedy_via_llm(
    current_subgoal: str,
    segment: Sequence[TypedFeedback],
    llm_chat,
    deadlock_signal: Mapping | None = None,
) -> KnowledgeEntry | None:
    """Use the planner LLM to produce a remedy JSON, then wrap as KnowledgeEntry."""

    if not segment and not deadlock_signal:
        return None
    feedback_dicts = [e.to_dict() for e in segment]
    user = render_remedy_user_prompt(
        current_subgoal=current_subgoal,
        recent_segment=feedback_dicts,
        deadlock_signal=deadlock_signal,
    )
    raw = llm_chat(REMEDY_SYSTEM_PROMPT, user)
    if not isinstance(raw, Mapping):
        return None

    repair = list(raw.get("repair_action") or [])
    if not repair:
        return None

    failure_type = str(
        raw.get("failure_type")
        or (segment[-1].f if segment else None)
        or (deadlock_signal.get("failure_type") if deadlock_signal else None)
        or "UNKNOWN"
    )
    scope = _normalize_scope(raw.get("scope"))
    category = _normalize_category(raw.get("category"))

    # Override: a deadlock prior should not be downgraded to subgoal_local.
    if deadlock_signal and category == RemedyCategory.GENERIC:
        category = RemedyCategory.DEADLOCK_PATTERN
        scope = RemedyScope.TASK_GLOBAL

    return KnowledgeEntry(
        type="remedy",
        c=list(raw.get("trigger_context") or []),
        u={
            "risk_pattern": str(raw.get("risk_pattern") or ""),
            "repair_action": repair,
            "scope": scope,
            "category": category,
            "current_subgoal": current_subgoal,
        },
        phi={
            "applicability": list(raw.get("applicability") or []),
        },
        rho=float(raw.get("confidence", 0.6)),
        E=[e.feedback_id for e in segment],
        failure_type=failure_type,
    )


def induce_remedy_if_needed(
    current_subgoal: str,
    recent_segment: Sequence[TypedFeedback],
    llm_chat,
    eta_fail: float = 0.5,
    deadlock_signal: Mapping | None = None,
) -> KnowledgeEntry | None:
    """Algorithm 2 stage 3 for remedies + Eq. (5) trigger + Figure 3 deadlock."""

    if not need_remedy(recent_segment, eta_fail=eta_fail, deadlock_signal=deadlock_signal):
        return None
    return build_remedy_via_llm(
        current_subgoal=current_subgoal,
        segment=recent_segment,
        llm_chat=llm_chat,
        deadlock_signal=deadlock_signal,
    )
