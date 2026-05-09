"""Typed execution feedback (MineEvolve paper Eq. 1).

A subgoal's execution is compressed into a single ``TypedFeedback`` object:

    e_i = (z_i, y_i, Δs_i, Δv_i, f_i, p_i, ℓ_i)

Each field is matchable, verifiable, and directly usable by downstream
modules (Inducer, Curator, Adaptor) - which is what distinguishes MineEvolve's
typed feedback from free-form trajectory reflection.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TypedFeedback:
    """The structured execution feedback for one subgoal (paper Eq. 1)."""

    z: str                                    # subgoal description
    y: int = 0                                # success label in {0, 1}
    delta_s: Dict[str, Any] = field(default_factory=dict)   # state change Δs_i
    delta_v: Dict[str, int] = field(default_factory=dict)   # inventory change Δv_i
    f: str = "NONE"                           # failure type label
    p: float = 0.0                            # progress score (Eq. 2)
    l: bool = False                           # stagnation flag (Eq. 3)

    # Bookkeeping (not part of the paper's tuple but needed for storage)
    feedback_id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:8]}")
    subgoal_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    coords: List[float] = field(default_factory=list)
    note: str = ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "subgoal_id": self.subgoal_id,
            "z": self.z,
            "y": int(self.y),
            "delta_s": dict(self.delta_s),
            "delta_v": dict(self.delta_v),
            "f": self.f,
            "p": float(self.p),
            "l": bool(self.l),
            "coords": list(self.coords),
            "note": self.note,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedFeedback":
        return cls(
            feedback_id=str(data.get("feedback_id") or f"e_{uuid.uuid4().hex[:8]}"),
            subgoal_id=data.get("subgoal_id"),
            z=str(data.get("z", "")),
            y=int(data.get("y", 0)),
            delta_s=dict(data.get("delta_s") or {}),
            delta_v={str(k): int(v) for k, v in (data.get("delta_v") or {}).items()},
            f=str(data.get("f", "NONE")),
            p=float(data.get("p", 0.0)),
            l=bool(data.get("l", False)),
            coords=list(data.get("coords") or []),
            note=str(data.get("note", "")),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def is_successful(self) -> bool:
        return int(self.y) == 1

    def is_stagnant(self) -> bool:
        return bool(self.l)
