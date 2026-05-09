"""Monitor module (paper Section 3.2): typed execution feedback extraction.

The ``Monitor`` class coordinates the four primitives in Algorithm 2 stage 1:
ParseTrace -> InventoryDiff -> StateDiff -> CheckSuccess -> DiagnoseFailure
-> ProgressScore -> DetectStagnation, and emits a TypedFeedback into the
episode buffer.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .buffer import FeedbackBuffer
from .failure import FAILURE_TYPES, diagnose_failure
from .feedback import TypedFeedback
from .progress import ProgressWeights, detect_stagnation, progress_score

__all__ = [
    "FeedbackBuffer",
    "FAILURE_TYPES",
    "Monitor",
    "ProgressWeights",
    "TypedFeedback",
    "detect_stagnation",
    "diagnose_failure",
    "progress_score",
]


class Monitor:
    """Compress raw env traces into TypedFeedback (paper Eq. 1)."""

    def __init__(self, weights: ProgressWeights | None = None) -> None:
        self.weights = weights or ProgressWeights()

    def build_feedback(
        self,
        subgoal: Tuple[str, int],
        target_item: str | None,
        coords_history: Sequence[Sequence[float]],
        delta_v: Mapping[str, int],
        delta_s: Mapping[str, Any],
        gui_open: bool,
        success: int,
        timed_out: bool = False,
        died: bool = False,
        p_goal: float = 0.0,
        subgoal_id: str | None = None,
        coords_now: Sequence[float] | None = None,
        note: str = "",
    ) -> TypedFeedback:
        p = progress_score(coords_history, delta_v, p_goal, weights=self.weights)
        l = detect_stagnation(p, success, weights=self.weights)
        f = diagnose_failure(
            success=success,
            coords_history=coords_history,
            delta_v=delta_v,
            gui_open=gui_open,
            target_item=target_item,
            timed_out=timed_out,
            died=died,
        )
        z = subgoal[0] if isinstance(subgoal, tuple) else str(subgoal)
        return TypedFeedback(
            z=z,
            y=int(success),
            delta_s=dict(delta_s),
            delta_v={str(k): int(v) for k, v in dict(delta_v).items()},
            f=f,
            p=float(p),
            l=bool(l),
            subgoal_id=subgoal_id,
            coords=list(coords_now or []),
            note=note,
        )
