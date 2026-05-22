"""Custom MineRL environment specs for MineEvolve.

Defines a single parameterised env spec used by all seven tech-tree task groups
(Wooden, Stone, Iron, Gold, Redstone, Diamond, Armor). The spec inherits from
MineRL's HumanSurvival and:

  * forbids any non-empty initial inventory (StrictEmptyInventoryStart);
  * exposes a ChatAction so the wrapper can issue game commands at runtime
    (gamerule, time, ore spawn near the agent's feet, etc.);
  * applies quality-of-life initial conditions (always day, allow spawning,
    do not pass real time);
  * is biased to a configurable preferred biome (e.g. forest for wooden
    tasks, plains for stone/iron tasks).

This is a clean rewrite by us; no source files are copied from MCU or
Optimus-1.
"""

from __future__ import annotations

import json
from typing import List, Sequence

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.mc import INVERSE_KEYMAP

from .chat_action import ChatAction
from .inventory_agent_start import StrictEmptyInventoryStart


MINUTE = 1200  # MineRL game ticks per simulated minute


class MineEvolveBaseSpec(HumanSurvival):
    """Base env spec used by every tech-tree benchmark group."""

    def __init__(
        self,
        env_name: str,
        prefer_biome: str = "plains",
        max_minutes: int = 10,
        seed: int | None = None,
    ) -> None:
        self._env_name = env_name
        self._prefer_biome = prefer_biome
        self._max_minutes = max_minutes
        self._seed = seed
        super().__init__(
            name=env_name,
            max_episode_steps=-1 if max_minutes < 0 else max_minutes * MINUTE,
            preferred_spawn_biome=prefer_biome,
        )

    # ------------------------------------------------------------------
    # Server-side world / initial conditions
    # ------------------------------------------------------------------

    def create_server_world_generators(self) -> List[Handler]:
        generator_options = {}
        if self._seed is not None:
            generator_options["seed"] = str(self._seed)
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True,
                generator_options=json.dumps(generator_options),
            )
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    # ------------------------------------------------------------------
    # Agent-side observation / action / start
    # ------------------------------------------------------------------

    def create_agent_start(self) -> List[Handler]:
        # Empty inventory is enforced inside StrictEmptyInventoryStart.
        base: Sequence[Handler] = super().create_agent_start()
        return list(base) + [
            StrictEmptyInventoryStart([]),
            handlers.PreferredSpawnBiome(self._prefer_biome),
            handlers.DoneOnDeath(),
        ]

    def create_actionables(self) -> List[Handler]:
        return list(super().create_actionables()) + [ChatAction()]

    # ------------------------------------------------------------------
    # Helpers used by other modules
    # ------------------------------------------------------------------

    @staticmethod
    def keymap() -> dict:
        return dict(INVERSE_KEYMAP)


def register_mineevolve_env(
    env_name: str,
    prefer_biome: str = "plains",
    max_minutes: int = 10,
    seed: int | None = None,
) -> None:
    """Register the env with Gym so callers can `gym.make(env_name)`."""

    spec = MineEvolveBaseSpec(
        env_name=env_name,
        prefer_biome=prefer_biome,
        max_minutes=max_minutes,
        seed=seed,
    )
    spec.register()
