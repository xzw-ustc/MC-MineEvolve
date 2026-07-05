"""Failure type diagnosis for typed execution feedback.

Closed vocabulary of failure labels used by Inducer/Curator. The labels
mirror the four "Failure / Stagnation Signals" in paper Figure 3
(Navigation Stuck, Target Unreachable, GUI Blockage, Missing Tool) plus
auxiliary labels for death and timeout. Keeping the vocabulary discrete
lets Curator validation (Eq. 6) and Adaptor retrieval check matchability
against the current state.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..util.items import inventory_delta_satisfies


# Closed vocabulary for f_i in Eq. (1). The first five entries align with
# paper Figure 3's Monitor "Failure / Stagnation Signals".
FAILURE_TYPES: tuple[str, ...] = (
    "NONE",
    "NAV_STUCK",          # repeated movement near same position, no inv gain
    "TARGET_UNREACHABLE", # agent moved a lot but the target item never gained
    "GUI_FAIL",           # crafting/inventory GUI did not produce result
    "MISSING_ITEM",       # missing precondition (tool, table, fuel, ...)
    "RECIPE_FAIL",        # recipe attempted but yields no item
    "SMELT_FAIL",         # smelting attempted but yields no item
    "DEADLOCK",           # cross-subgoal repeating same failure type
    "MOB_KILLED",         # agent died during execution
    "TIMEOUT",            # subgoal exceeded its per-step budget
    "UNKNOWN",
)


# Coordinate-variance threshold separating NAV_STUCK (low) from
# TARGET_UNREACHABLE (high) under the same "no inv gain" symptom.
LOW_MOVEMENT_THRESHOLD: float = 1.0
HIGH_MOVEMENT_THRESHOLD: float = 8.0


def diagnose_failure(
    success: int,
    coords_history: Sequence[Sequence[float]],
    delta_v: Mapping[str, int],
    gui_open: bool,
    target_item: str | None,
    timed_out: bool = False,
    died: bool = False,
) -> str:
    """Return one of FAILURE_TYPES based on observed signals.

    Diagnosis order (most specific first):
      1. success     -> NONE
      2. died        -> MOB_KILLED
      3. inv increased on target -> NONE
      4. GUI open + no inv change -> GUI_FAIL
      5. no inv change + low coord movement (stuck pattern) -> NAV_STUCK
      6. no inv change + high coord movement -> TARGET_UNREACHABLE
      7. no inv change + a target was named -> MISSING_ITEM
      8. timed out (and none above matched) -> TIMEOUT
      9. UNKNOWN
    """

    if int(success) == 1:
        return "NONE"
    if died:
        return "MOB_KILLED"

    inv_increased_target = target_item is not None and inventory_delta_satisfies(
        delta_v,
        target=str(target_item),
        n=1,
    )
    if inv_increased_target:
        return "NONE"

    if gui_open and not delta_v:
        return "GUI_FAIL"

    span = _coord_span(coords_history)
    if not delta_v:
        if span < LOW_MOVEMENT_THRESHOLD:
            return "NAV_STUCK"
        if span > HIGH_MOVEMENT_THRESHOLD:
            return "TARGET_UNREACHABLE"

    if target_item is not None and int(delta_v.get(str(target_item), 0)) == 0:
        return "MISSING_ITEM"

    if timed_out:
        return "TIMEOUT"

    return "UNKNOWN"


def _coord_span(coords: Sequence[Sequence[float]]) -> float:
    """Sum of per-axis ranges across the coordinate window."""

    if len(coords) < 2:
        return 0.0
    dim = max((len(c) for c in coords), default=0)
    if dim == 0:
        return 0.0
    span = 0.0
    for axis in range(dim):
        vals = [float(c[axis]) for c in coords if len(c) > axis]
        if len(vals) < 2:
            continue
        span += max(vals) - min(vals)
    return span


def is_low_movement(coords: Sequence[Sequence[float]], threshold: float = LOW_MOVEMENT_THRESHOLD) -> bool:
    """Backwards-compatible helper used by older callers."""

    return _coord_span(coords) < float(threshold)
