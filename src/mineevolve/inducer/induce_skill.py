"""Skill induction: successful feedback segments -> KnowledgeEntry(skill).

Implements Algorithm 2 stage 3 (lines 15-21) for the skill branch:

    for B+ in S+:
        if PassTargetCheck(B+):
            k+ = BuildSkill(B+)
            E(k+) = B+
            C = C union {k+}
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from ..monitor.feedback import TypedFeedback
from .prompts import (
    SKILL_SYSTEM_PROMPT,
    render_skill_user_prompt,
)
from .schema import KnowledgeEntry


def pass_target_check(segment: Sequence[TypedFeedback]) -> bool:
    """Check whether a successful segment ends with an inventory-positive
    final feedback - the simplest verifiable target check."""

    if not segment:
        return False
    if not all(e.is_successful() for e in segment):
        return False
    last = segment[-1]
    if not last.delta_v:
        return False
    return any(int(v) > 0 for v in last.delta_v.values())


def build_skill_via_llm(
    task_goal: str,
    segment: Sequence[TypedFeedback],
    llm_chat,
) -> KnowledgeEntry | None:
    """Use the planner LLM to produce a skill JSON, then wrap as KnowledgeEntry.

    ``llm_chat`` is a callable: (system: str, user: str) -> dict | None
    The callable is responsible for JSON parsing and retries.
    """

    if not segment:
        return None
    feedback_dicts = [e.to_dict() for e in segment]
    user = render_skill_user_prompt(task_goal=task_goal, successful_segment=feedback_dicts)
    raw = llm_chat(SKILL_SYSTEM_PROMPT, user)
    if not isinstance(raw, Mapping):
        return None

    steps = list(raw.get("steps") or [])
    if not steps:
        return None

    return KnowledgeEntry(
        type="skill",
        c=list(raw.get("trigger_context") or []),
        u={
            "preconditions": list(raw.get("preconditions") or []),
            "steps": steps,
            "observed_effects": list(raw.get("observed_effects") or []),
            "task_goal": task_goal,
        },
        phi={
            "verification": dict(raw.get("verification") or {}),
        },
        rho=float(raw.get("confidence", 0.6)),
        E=[e.feedback_id for e in segment],
        failure_type="NONE",
    )


def induce_skills(
    task_goal: str,
    segments: Iterable[Sequence[TypedFeedback]],
    llm_chat,
) -> List[KnowledgeEntry]:
    """Algorithm 2 stage 3 for skills."""

    out: List[KnowledgeEntry] = []
    for seg in segments:
        if not pass_target_check(seg):
            continue
        entry = build_skill_via_llm(task_goal, seg, llm_chat)
        if entry is not None:
            out.append(entry)
    return out
