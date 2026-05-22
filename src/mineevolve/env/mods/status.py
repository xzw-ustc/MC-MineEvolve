"""StatusMod: track agent state (position, health, hunger, GUI, inventory).

Exposes a structured snapshot consumed by Monitor (Eq. 1: state change Δs and
inventory change Δv) and by the wrapper (auto-pickaxe selection).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Tuple


class StatusMod:
    """Maintains a rolling snapshot of agent state across env steps."""

    def __init__(self, cfg: Mapping[str, Any], logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger("mineevolve.env.status")
        self._coord_window_size = 256
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._inventory: Dict[str, int] = {}
        self._plain_inventory: Dict[int, Dict[str, Any]] = {}
        self._coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._coord_history: Deque[Tuple[float, float, float]] = deque(
            maxlen=self._coord_window_size
        )
        self._health: float = 20.0
        self._hunger: float = 20.0
        self._gui_open: bool = False
        self._inventory_changed: bool = False
        self._inventory_diff_last_step: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Step update
    # ------------------------------------------------------------------

    def step(self, obs: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        self._gui_open = bool(obs.get("isGuiOpen", False))

        loc = obs.get("location_stats") or {}
        try:
            self._coords = (
                float(loc.get("xpos", self._coords[0])),
                float(loc.get("ypos", self._coords[1])),
                float(loc.get("zpos", self._coords[2])),
            )
        except (TypeError, ValueError):
            pass
        self._coord_history.append(self._coords)

        try:
            self._health = float(loc.get("life", self._health) or self._health)
        except (TypeError, ValueError):
            pass
        try:
            self._hunger = float(loc.get("food", self._hunger) or self._hunger)
        except (TypeError, ValueError):
            pass

        new_inventory = self._normalize_inventory(obs.get("inventory", {}))
        self._inventory_diff_last_step = self._diff_inventory(self._inventory, new_inventory)
        self._inventory_changed = bool(self._inventory_diff_last_step)
        self._inventory = new_inventory

        plain = obs.get("plain_inventory")
        if isinstance(plain, Mapping):
            self._plain_inventory = {int(k): dict(v) for k, v in plain.items() if isinstance(v, Mapping)}

    # ------------------------------------------------------------------
    # Snapshot accessors used by other modules
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "coords": list(self._coords),
            "ypos": self._coords[1],
            "health": self._health,
            "hunger": self._hunger,
            "isGuiOpen": self._gui_open,
            "inventory_changed": self._inventory_changed,
            "inventory_diff": dict(self._inventory_diff_last_step),
            "inventory": dict(self._inventory),
        }

    def get_height(self) -> int:
        return int(self._coords[1])

    def coords(self) -> Tuple[float, float, float]:
        return self._coords

    def coord_history(self) -> List[Tuple[float, float, float]]:
        return list(self._coord_history)

    def inventory(self) -> Dict[str, int]:
        return dict(self._inventory)

    def plain_inventory(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._plain_inventory)

    @property
    def inventory_changed(self) -> bool:
        return self._inventory_changed

    def inventory_diff(self) -> Dict[str, int]:
        return dict(self._inventory_diff_last_step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_inventory(raw: Any) -> Dict[str, int]:
        if not isinstance(raw, Mapping):
            return {}
        out: Dict[str, int] = {}
        for k, v in raw.items():
            try:
                count = int(v)
            except (TypeError, ValueError):
                try:
                    count = int(getattr(v, "item", lambda: 0)())
                except Exception:
                    continue
            if count > 0:
                out[str(k)] = count
        return out

    @staticmethod
    def _diff_inventory(prev: Mapping[str, int], curr: Mapping[str, int]) -> Dict[str, int]:
        diff: Dict[str, int] = {}
        keys = set(prev) | set(curr)
        for k in keys:
            d = int(curr.get(k, 0)) - int(prev.get(k, 0))
            if d != 0:
                diff[k] = d
        return diff
