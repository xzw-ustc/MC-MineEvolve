"""TaskCheckerMod: inventory-delta based subgoal completion check.

A subgoal in MineEvolve is a tuple ``(target_item: str, n: int)``. We compare
the *increase* in inventory between the start of a subgoal and the current
step. The subgoal is marked complete the first time the agent gains >= n
units of ``target_item``.

The wrapper resets us via ``reset(inventory)`` whenever a new subgoal begins.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ...util.items import inventory_delta_satisfies


class TaskCheckerMod:
    """Track inventory delta against a (item, count) target."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.reset({})

    def reset(self, inventory: Mapping[str, Any] | None = None) -> None:
        self._baseline = self._normalize(inventory or {})

    @staticmethod
    def _normalize(inventory: Mapping[str, Any]) -> dict:
        out: dict = {}
        for k, v in (inventory or {}).items():
            try:
                count = int(v)
            except (TypeError, ValueError):
                continue
            if count > 0:
                out[str(k)] = count
        return out

    def step(
        self,
        inventory: Mapping[str, Any],
        subgoal: Tuple[str, int] | None,
    ) -> bool:
        if subgoal is None:
            return False
        target, need = subgoal
        if not target or need <= 0:
            return True
        current = self._normalize(inventory)
        delta = {
            item: int(current.get(item, 0)) - int(self._baseline.get(item, 0))
            for item in set(current) | set(self._baseline)
        }
        return inventory_delta_satisfies(delta, target=target, n=need)
