"""Item matching helpers for Minecraft inventory checks."""

from __future__ import annotations

from typing import Mapping


LOG_TARGETS: tuple[str, ...] = ("log", "oak_log", "wood")


def item_matches_target(item: str, target: str) -> bool:
    """Return True when an inventory item satisfies a target item.

    Exact matching is used by default.  For wood-gathering tasks, planners may
    ask for ``oak_log`` while the forest seed exposes dark-oak trees; in that
    case any ``*_log`` is valid evidence for the higher-level "collect wood"
    skill/remedy loop.
    """

    item_norm = _norm(item)
    target_norm = _norm(target)
    if not item_norm or not target_norm:
        return False
    if item_norm == target_norm:
        return True
    if target_norm in LOG_TARGETS and item_norm.endswith("_log"):
        return True
    return False


def inventory_delta_satisfies(delta_v: Mapping[str, int], target: str, n: int = 1) -> bool:
    total = 0
    for item, delta in delta_v.items():
        if item_matches_target(str(item), target):
            total += int(delta)
    return total >= int(n)


def inventory_satisfies(inventory: Mapping[str, int], target: str, n: int = 1) -> bool:
    total = 0
    for item, count in inventory.items():
        if item_matches_target(str(item), target):
            total += int(count)
    return total >= int(n)


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace("minecraft:", "")
