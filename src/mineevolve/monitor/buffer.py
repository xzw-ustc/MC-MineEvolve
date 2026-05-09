"""Episode-level feedback buffer B (paper Algorithm 1, line 3).

Append-only list of TypedFeedback items, with helpers used by Inducer to
identify successful and failed segments (Algorithm 2, lines 13-14) and to
detect cross-subgoal deadlock patterns (paper Figure 3, "Deadlock Pattern"
under task-level remedy induction).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from .feedback import TypedFeedback


class FeedbackBuffer:
    """The buffer B accumulated within a single episode."""

    def __init__(self) -> None:
        self._items: List[TypedFeedback] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._items = []

    def append(self, e: TypedFeedback) -> None:
        self._items.append(e)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def items(self) -> List[TypedFeedback]:
        return list(self._items)

    def latest(self) -> TypedFeedback | None:
        return self._items[-1] if self._items else None

    def recent(self, n: int) -> List[TypedFeedback]:
        if n <= 0:
            return []
        return list(self._items[-n:])

    def to_jsonable(self) -> List[dict]:
        return [e.to_dict() for e in self._items]

    # ------------------------------------------------------------------
    # Segmentation helpers (Algorithm 2 lines 13-14)
    # ------------------------------------------------------------------

    def successful_segments(self, min_len: int = 1) -> List[List[TypedFeedback]]:
        """Maximal runs of consecutive successful TypedFeedbacks."""

        out: List[List[TypedFeedback]] = []
        cur: List[TypedFeedback] = []
        for e in self._items:
            if e.is_successful():
                cur.append(e)
            else:
                if len(cur) >= min_len:
                    out.append(cur)
                cur = []
        if len(cur) >= min_len:
            out.append(cur)
        return out

    def failure_or_stagnation_segments(self, window: int) -> List[List[TypedFeedback]]:
        """Recent failure / stagnation windows used for remedy induction.

        We return at most one window: the most recent ``window`` items, but
        only if either the failure rate >= eta_fail (handled by caller via
        Eq. 5) or the latest item is stagnant. The caller decides whether to
        induce a remedy from it.
        """

        if not self._items:
            return []
        return [self.recent(window)]

    def failure_rate(self, window: int) -> float:
        if window <= 0 or not self._items:
            return 0.0
        seg = self.recent(window)
        if not seg:
            return 0.0
        return float(sum(1 for e in seg if not e.is_successful())) / float(len(seg))

    # ------------------------------------------------------------------
    # Deadlock detection (Figure 3: task-level "Deadlock Pattern" trigger)
    # ------------------------------------------------------------------

    def detect_deadlock(
        self,
        window: int = 8,
        min_distinct_subgoals: int = 2,
        min_repeats: int = 3,
    ) -> Dict[str, int] | None:
        """Detect a cross-subgoal repeating-failure pattern.

        Returns ``{"failure_type": str, "count": int, "n_distinct_subgoals": int}``
        if the same non-NONE failure type appears at least ``min_repeats``
        times across at least ``min_distinct_subgoals`` distinct subgoals
        within the recent window. Otherwise returns ``None``.

        This is the signal that distinguishes a task-level deadlock from a
        purely local subgoal-level stagnation (which Eq. 3 already covers).
        """

        if window <= 0:
            return None
        seg = self.recent(window)
        if not seg:
            return None

        per_type: Dict[str, set] = {}
        per_type_count: Counter = Counter()
        for e in seg:
            if e.is_successful() or not e.f or e.f == "NONE":
                continue
            per_type.setdefault(e.f, set()).add(e.subgoal_id or e.z)
            per_type_count[e.f] += 1

        for failure_type, subgoal_ids in per_type.items():
            count = int(per_type_count[failure_type])
            n_distinct = len(subgoal_ids)
            if count >= int(min_repeats) and n_distinct >= int(min_distinct_subgoals):
                return {
                    "failure_type": str(failure_type),
                    "count": count,
                    "n_distinct_subgoals": n_distinct,
                }
        return None
