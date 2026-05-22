"""Strict empty-inventory agent-start handler.

Per MineEvolve paper Section 4.1 and Appendix B.2, every episode must start
with an empty inventory; all task-required resources are obtained through
interaction with the environment. We deliberately keep this handler minimal
and never expose any /give-style cheating shortcut.
"""

from __future__ import annotations

from typing import Sequence

from minerl.herobraine.hero import handlers


class StrictEmptyInventoryStart(handlers.InventoryAgentStart):
    """Initial inventory handler that enforces an empty-inventory contract.

    The constructor accepts an optional sequence (kept for API symmetry with
    MineRL's parent handler), but if a non-empty inventory is supplied we raise
    a ValueError. This guards against accidental reintroduction of
    cheat-style item grants in benchmark configs.
    """

    def __init__(self, inventory: Sequence[dict] = ()) -> None:
        if len(inventory) > 0:
            raise ValueError(
                "MineEvolve forbids non-empty initial inventories. "
                "All required resources must be obtained through interaction. "
                f"Received: {list(inventory)!r}"
            )
        super().__init__([])
