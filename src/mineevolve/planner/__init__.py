"""High-level LLM planner for MineEvolve."""

from __future__ import annotations

from typing import Mapping, Sequence

from .backends import PlannerBackend, get_backend
from .base import Plan, PlannerBackend as _PlannerBackend  # re-export
from .base import Subgoal
from .parser import parse_plan
from .prompts import PLANNER_SYSTEM_PROMPT, render_planner_user_prompt


def make_chat_callable(backend: PlannerBackend, default_max_tokens: int = 1024):
    """Adapter producing a (system, user) -> dict | None callable.

    Used by Inducer / Adaptor which only need JSON output, not raw text.
    """

    from ..util.prompt import parse_json

    def _chat(system: str, user: str) -> dict | None:
        text = backend.chat(system=system, user=user, max_tokens=default_max_tokens)
        result = parse_json(text)
        return result if isinstance(result, dict) else None

    return _chat


def generate_initial_plan(
    backend: PlannerBackend,
    task_goal: str,
    state: Mapping,
    completed_prefix: Sequence[str] = (),
    retrieved_skills: Sequence[Mapping] = (),
    active_remedies: Sequence[Mapping] = (),
    max_tokens: int = 1024,
) -> Plan | None:
    """One-shot planner call returning a parsed Plan."""

    user = render_planner_user_prompt(
        task_goal=task_goal,
        state=state,
        completed_prefix=completed_prefix,
        retrieved_skills=retrieved_skills,
        active_remedies=active_remedies,
    )
    text = backend.chat(system=PLANNER_SYSTEM_PROMPT, user=user, max_tokens=max_tokens)
    return parse_plan(text, fallback_goal=task_goal)


__all__ = [
    "Plan",
    "Subgoal",
    "PlannerBackend",
    "get_backend",
    "make_chat_callable",
    "generate_initial_plan",
    "parse_plan",
    "PLANNER_SYSTEM_PROMPT",
    "render_planner_user_prompt",
]
