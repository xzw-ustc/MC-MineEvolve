"""Knowledge-conditioned local repair (paper Eq. 8 + Algorithm 3 stage 3).

    z'_{1:N} = [ z_{1:i}, Repair(g, s_i, z_{1:i}, K^R_i, A_i) ]

The Adaptor freezes the completed-or-still-valid prefix and asks the LLM
to regenerate ONLY the unfinished suffix under the retrieved knowledge and
the active remedy set.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from ..inducer.schema import KnowledgeEntry
from ..monitor.feedback import TypedFeedback
from .prompts import ADAPTOR_SYSTEM_PROMPT, render_adaptor_user_prompt


# ----------------------------------------------------------------------
# Prefix freezing
# ----------------------------------------------------------------------

def freeze_prefix(
    plan: Sequence[Mapping[str, Any]],
    next_index: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Split the plan at ``next_index`` into (kept_prefix, suffix_to_repair)."""

    if next_index <= 0:
        return [], list(plan)
    if next_index >= len(plan):
        return list(plan), []
    return list(plan[:next_index]), list(plan[next_index:])


def need_repair(
    recent_feedback: Sequence[TypedFeedback],
    eta_fail: float = 0.5,
) -> bool:
    """Same trigger as remedy generation (Eq. 5)."""

    if not recent_feedback:
        return False
    if recent_feedback[-1].is_stagnant():
        return True
    failures = sum(1 for e in recent_feedback if not e.is_successful())
    return float(failures) / float(len(recent_feedback)) >= float(eta_fail)


# ----------------------------------------------------------------------
# Repair via LLM
# ----------------------------------------------------------------------

def repair_plan(
    task_goal: str,
    state: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    next_index: int,
    recent_feedback: Sequence[TypedFeedback],
    retrieved_skills: Sequence[KnowledgeEntry],
    active_remedies: Sequence[KnowledgeEntry],
    llm_chat,
) -> Mapping[str, Any] | None:
    """Algorithm 3 stage 3: replace plan[next_index:] with LLM-repaired suffix.

    Returns the parsed JSON repair-output dict (with kept_prefix + repaired_suffix),
    or None if the LLM call failed.
    """

    kept_prefix, _ = freeze_prefix(plan, next_index)
    user = render_adaptor_user_prompt(
        task_goal=task_goal,
        state=state,
        current_plan=plan,
        frozen_prefix_ids=[str(p.get("subgoal_id") or "") for p in kept_prefix],
        recent_feedback=[e.to_dict() for e in recent_feedback],
        retrieved_skills=[k.to_dict() for k in retrieved_skills],
        active_remedies=[k.to_dict() for k in active_remedies],
    )
    raw = llm_chat(ADAPTOR_SYSTEM_PROMPT, user)
    if not isinstance(raw, Mapping):
        return None
    if "repaired_suffix" not in raw:
        return None
    return raw


def assemble_repaired_plan(
    plan: Sequence[Mapping[str, Any]],
    next_index: int,
    repair_output: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Stitch frozen prefix and repaired suffix back into a single plan."""

    kept_prefix, _ = freeze_prefix(plan, next_index)
    suffix = list(repair_output.get("repaired_suffix") or [])
    return list(kept_prefix) + suffix
