"""Parse LLM planner / adaptor JSON output into typed dataclasses."""

from __future__ import annotations

import uuid
from typing import Mapping

from ..util.prompt import parse_json
from .base import Plan, Subgoal


def parse_plan(text: str, fallback_goal: str = "") -> Plan | None:
    """Parse planner output text into a Plan, with light auto-fixing."""

    raw = parse_json(text)
    if not isinstance(raw, Mapping):
        return None
    plan = Plan.from_dict(raw)
    if not plan.plan_id:
        plan.plan_id = f"p_{uuid.uuid4().hex[:8]}"
    # Auto-assign subgoal ids if the LLM forgot
    for i, sg in enumerate(plan.subgoals):
        if not sg.subgoal_id:
            sg.subgoal_id = f"sg_{i + 1:03d}"
    if not plan.subgoals and fallback_goal:
        plan.subgoals.append(
            Subgoal(
                subgoal_id="sg_001",
                condition=fallback_goal,
                task_kind="mine",
                executor_hint="stevei",
                mode="move",
                timeout_s=60,
                checks=[],
                rationale="fallback single-step plan",
            )
        )
    return plan
