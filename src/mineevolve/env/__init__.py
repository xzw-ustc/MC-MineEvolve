"""Environment layer for MineEvolve.

Public helpers:
  * make_env(cfg, logger) - build a registered MineRL env wrapped with
    MineEvolveEnvWrapper (dynamic ore spawn + auto pickaxe + stuck recovery).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gym

from .custom_env import register_mineevolve_env
from .wrapper import MineEvolveEnvWrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_env(cfg: "DictConfig", logger: logging.Logger | None = None) -> MineEvolveEnvWrapper:
    """Construct a fully wrapped MineEvolve env from a Hydra config.

    Expected Hydra config shape:
        benchmark:
          env:
            name: WoodenTaskEnv-v0
            prefer_biome: forest
            max_minutes: 2
        commands:
          - /gamerule keepInventory true
          - ...
    """

    logger = logger or logging.getLogger("mineevolve.env")
    env_cfg = cfg.get("benchmark", {}).get("env", cfg.get("env", {}))
    if not env_cfg:
        raise KeyError("Missing env config: expected cfg.benchmark.env or cfg.env")

    register_mineevolve_env(
        env_name=str(env_cfg["name"]),
        prefer_biome=str(env_cfg.get("prefer_biome", "plains")),
        max_minutes=int(env_cfg.get("max_minutes", 10)),
        seed=env_cfg.get("seed", None),
    )

    raw = gym.make(str(env_cfg["name"]))
    return MineEvolveEnvWrapper(raw, cfg=cfg, logger=logger)


__all__ = ["make_env", "MineEvolveEnvWrapper", "register_mineevolve_env"]
