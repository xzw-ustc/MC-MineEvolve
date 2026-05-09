"""Knowledge entry schema (paper Eq. 4 + Figure 3).

A unified knowledge entry is a 6-tuple:

    k = (type, c, u, phi, rho, E)        type in {skill, remedy}

where
    type : "skill" | "remedy"
    c    : trigger context (list of strings)
    u    : knowledge content (skill steps OR remedy repair-action dict)
    phi  : verification or applicability condition
    rho  : confidence score in [0, 1]
    E    : list of supporting feedback ids

For remedies, ``u`` additionally carries a two-level scope/category that
mirrors paper Figure 3:

    scope    in {SUBGOAL_LOCAL, UNFINISHED_PLAN_SUFFIX, TASK_GLOBAL}
    category in {GENERIC,                                     # default subgoal-level
                 MISSING_PREREQUISITE,                        # task-level
                 DEADLOCK_PATTERN,                            # task-level
                 RECOVERY_STRATEGY}                           # task-level
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


KnowledgeType = Literal["skill", "remedy"]


# ----------------------------------------------------------------------
# Two-level remedy enums (paper Figure 3)
# ----------------------------------------------------------------------


class RemedyScope:
    """Where the remedy applies in the plan (Figure 3 + Eq. 8)."""

    SUBGOAL_LOCAL = "subgoal_local"                # only modifies the immediate next subgoal
    UNFINISHED_PLAN_SUFFIX = "unfinished_plan_suffix"
    TASK_GLOBAL = "task_global"                    # may insert prerequisites at suffix head

    ALL: tuple[str, ...] = (SUBGOAL_LOCAL, UNFINISHED_PLAN_SUFFIX, TASK_GLOBAL)


class RemedyCategory:
    """Functional category of the remedy (Figure 3 task-level box)."""

    GENERIC = "generic"                            # default subgoal-level: trigger/risk/repair
    MISSING_PREREQUISITE = "missing_prerequisite"  # insert obtain/place X before retrying recipe
    DEADLOCK_PATTERN = "deadlock_pattern"          # break a cross-subgoal stuck loop
    RECOVERY_STRATEGY = "recovery_strategy"        # multi-step recovery sequence

    ALL: tuple[str, ...] = (
        GENERIC,
        MISSING_PREREQUISITE,
        DEADLOCK_PATTERN,
        RECOVERY_STRATEGY,
    )

    TASK_LEVEL: tuple[str, ...] = (
        MISSING_PREREQUISITE,
        DEADLOCK_PATTERN,
        RECOVERY_STRATEGY,
    )


@dataclass
class KnowledgeEntry:
    """The unified knowledge entry of paper Eq. (4)."""

    type: KnowledgeType
    c: List[str] = field(default_factory=list)              # trigger context
    u: Dict[str, Any] = field(default_factory=dict)         # content
    phi: Dict[str, Any] = field(default_factory=dict)       # verification / applicability
    rho: float = 0.5                                        # confidence
    E: List[str] = field(default_factory=list)              # supporting feedback ids

    # Bookkeeping for the external knowledge store
    knowledge_id: str = field(default_factory=lambda: f"k_{uuid.uuid4().hex[:8]}")
    support_count: int = 1
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    failure_type: str = "NONE"     # remedy convenience: f_i it addresses

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "type": self.type,
            "c": list(self.c),
            "u": dict(self.u),
            "phi": dict(self.phi),
            "rho": float(self.rho),
            "E": list(self.E),
            "support_count": int(self.support_count),
            "usage_count": int(self.usage_count),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "failure_type": str(self.failure_type),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            knowledge_id=str(data.get("knowledge_id") or f"k_{uuid.uuid4().hex[:8]}"),
            type=str(data.get("type", "skill")),  # type: ignore[arg-type]
            c=list(data.get("c") or []),
            u=dict(data.get("u") or {}),
            phi=dict(data.get("phi") or {}),
            rho=float(data.get("rho", 0.5)),
            E=list(data.get("E") or []),
            support_count=int(data.get("support_count", 1)),
            usage_count=int(data.get("usage_count", 0)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            failure_type=str(data.get("failure_type", "NONE")),
        )

    def is_skill(self) -> bool:
        return self.type == "skill"

    def is_remedy(self) -> bool:
        return self.type == "remedy"

    def short_summary(self) -> str:
        """Compact one-liner used in retrieval logs."""

        if self.is_skill():
            n_steps = len(self.u.get("steps", []) or [])
            return f"[skill {self.knowledge_id}] {n_steps} steps, rho={self.rho:.2f}"
        cat = str(self.u.get("category") or RemedyCategory.GENERIC)
        scope = str(self.u.get("scope") or RemedyScope.UNFINISHED_PLAN_SUFFIX)
        return (
            f"[remedy {self.knowledge_id}] {self.failure_type}/{cat}/{scope}, "
            f"rho={self.rho:.2f}"
        )

    # ------------------------------------------------------------------
    # Two-level remedy convenience
    # ------------------------------------------------------------------

    def is_task_level_remedy(self) -> bool:
        if not self.is_remedy():
            return False
        cat = str(self.u.get("category") or RemedyCategory.GENERIC)
        return cat in RemedyCategory.TASK_LEVEL

    def remedy_scope(self) -> str:
        return str(self.u.get("scope") or RemedyScope.UNFINISHED_PLAN_SUFFIX)

    def remedy_category(self) -> str:
        return str(self.u.get("category") or RemedyCategory.GENERIC)
