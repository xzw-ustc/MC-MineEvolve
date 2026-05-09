"""Planner base classes and subgoal schema (paper Appendix E)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


SUBGOAL_OUTPUT_SCHEMA: dict = {
    "plan_id": "p_xxxx",
    "subgoals": [
        {
            "subgoal_id": "sg_001",
            "condition": "short executable action",
            "task_kind": "mine|craft|smelt|use|combat|move|wait",
            "executor_hint": "stevei|mc_craft|mc_smelt|wait",
            "mode": "move|stay",
            "timeout_s": 60,
            "checks": [{"type": "inv_ge", "item": "<item>", "n": 1}],
            "rationale": "one short reason",
        }
    ],
    "global_constraints": [],
}


@dataclass
class Subgoal:
    """One executable subgoal (paper Appendix E planner output)."""

    subgoal_id: str
    condition: str
    task_kind: str = "mine"
    executor_hint: str = "stevei"
    mode: str = "move"
    timeout_s: int = 60
    checks: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""

    def primary_target(self) -> tuple[str, int] | None:
        """Return the inventory target (item, n) if a check expresses one."""

        for c in self.checks:
            if not isinstance(c, Mapping):
                continue
            if c.get("type") == "inv_ge":
                item = str(c.get("item") or "").strip()
                n = int(c.get("n") or 1)
                if item:
                    return item, n
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "condition": self.condition,
            "task_kind": self.task_kind,
            "executor_hint": self.executor_hint,
            "mode": self.mode,
            "timeout_s": int(self.timeout_s),
            "checks": list(self.checks),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Subgoal":
        return cls(
            subgoal_id=str(data.get("subgoal_id") or ""),
            condition=str(data.get("condition") or ""),
            task_kind=str(data.get("task_kind") or "mine"),
            executor_hint=str(data.get("executor_hint") or "stevei"),
            mode=str(data.get("mode") or "move"),
            timeout_s=int(data.get("timeout_s") or 60),
            checks=list(data.get("checks") or []),
            rationale=str(data.get("rationale") or ""),
        )


@dataclass
class Plan:
    plan_id: str
    subgoals: List[Subgoal] = field(default_factory=list)
    global_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "global_constraints": list(self.global_constraints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Plan":
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            subgoals=[Subgoal.from_dict(s) for s in (data.get("subgoals") or []) if isinstance(s, Mapping)],
            global_constraints=list(data.get("global_constraints") or []),
        )


# ----------------------------------------------------------------------
# LLM backend abstract interface
# ----------------------------------------------------------------------


class PlannerBackend:
    """All LLM backends must expose this single ``chat`` method."""

    name: str = "base"

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        raise NotImplementedError
