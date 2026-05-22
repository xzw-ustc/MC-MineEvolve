"""MineEvolve environment wrapper.

Adds two runtime helpers on top of a vanilla MineRL HumanSurvival env:

1. DynamicOreSpawnMixin
       When the agent is mining underground at an ore-bearing Y band, a small
       probability spawn of the matching ore is placed in the local block
       column via the chat /setblock command. The agent still has to navigate,
       break, and collect the block - this is NOT a /give shortcut.

2. AutoPickaxeMixin
       When the player is below Y=70 (i.e. mining), automatically switch the
       active hotbar slot to the highest-tier pickaxe currently held. STEVE-1
       does not reliably perform tool selection on its own, so this matches
       the experimental setup also used by prior systems.

We DELIBERATELY DO NOT implement an environment-level stuck recovery
(e.g. forced ``/kill`` after long Y-dwell). Such a mechanism would conflict
with the MineEvolve method itself: the paper's Monitor (Eq. 3) detects
exactly the same stagnation pattern (low progress + no inventory gain) and
the Inducer (Eq. 5) is supposed to convert it into a navigation remedy that
the Adaptor (Eq. 8) inserts into the unfinished plan (paper Section 3.3 and
Figure 7). Hard-killing the agent at the environment layer would steal that
signal, truncate the offending trace before it can be observed, and
contaminate the coordinate-variance term in ProgressScore (Eq. 2). The
episode horizon ``max_minutes`` is the only hard backstop.

The dwell counter is still tracked so the Monitor module can read it from
``StatusMod`` and use it inside Eq. (2)/(3) without imposing any
environment-level intervention.

Implementation is entirely original; the world-shaping ideas (Y-band ore
spawning, auto pickaxe) are inspired by published designs of recent
Minecraft agent systems but no source code is reused from any other repo.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import gym
import numpy as np

from .mods.recorder import RecorderMod
from .mods.status import StatusMod
from .mods.task_checker import TaskCheckerMod


# ----------------------------------------------------------------------
# Ore band configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OreBand:
    name: str
    y_min: int
    y_max: int
    max_count: int


DEFAULT_ORE_BANDS: Tuple[OreBand, ...] = (
    OreBand("coal_ore", y_min=45, y_max=50, max_count=6),
    OreBand("iron_ore", y_min=26, y_max=43, max_count=17),
    OreBand("gold_ore", y_min=17, y_max=26, max_count=10),
    OreBand("redstone_ore", y_min=1, y_max=16, max_count=12),
    OreBand("diamond_ore", y_min=1, y_max=14, max_count=14),
)

# Probability of spawning *any* ore at a given step when the agent enters a
# new Y position. Tuned to be sparse so the world feels mostly natural.
DEFAULT_SPAWN_PROBABILITY = 0.10
DEFAULT_VERTICAL_OFFSET_RANGE = (-5, -3)  # blocks below the agent's feet


@dataclass
class _OreState:
    """Mutable per-episode state for the dynamic-ore mixin."""

    spawned_at_y: Dict[int, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


class DynamicOreSpawnMixin:
    """Spawn ores near the agent at realistic Y bands."""

    bands: Sequence[OreBand] = DEFAULT_ORE_BANDS
    spawn_probability: float = DEFAULT_SPAWN_PROBABILITY
    vertical_offset_range: Tuple[int, int] = DEFAULT_VERTICAL_OFFSET_RANGE

    def _reset_ore_state(self) -> None:
        self._ore_state = _OreState()

    def _maybe_spawn_ore(self, env: gym.Env, ypos: int) -> None:
        if random.random() > self.spawn_probability:
            return
        if ypos in self._ore_state.spawned_at_y:
            return

        dy = random.randint(*self.vertical_offset_range)
        target_y = ypos + dy
        if target_y in self._ore_state.spawned_at_y:
            return

        for band in self.bands:
            if not (band.y_min <= ypos <= band.y_max):
                continue
            if not (band.y_min <= target_y <= band.y_max):
                continue
            if self._ore_state.counts.get(band.name, 0) >= band.max_count:
                continue

            self._ore_state.spawned_at_y[target_y] = band.name
            self._ore_state.counts[band.name] = (
                self._ore_state.counts.get(band.name, 0) + 1
            )
            self.execute_cmd(f"/setblock ~ ~{dy} ~ minecraft:{band.name}")
            return


# ----------------------------------------------------------------------
# Auto-pickaxe mixin
# ----------------------------------------------------------------------


class AutoPickaxeMixin:
    """Switch to the best available pickaxe whenever the agent is below Y=70."""

    pickaxe_priority: Tuple[str, ...] = (
        "diamond_pickaxe",
        "iron_pickaxe",
        "stone_pickaxe",
        "wooden_pickaxe",
    )
    auto_pickaxe_y_threshold: int = 70

    def _select_best_pickaxe_slot(
        self, ypos: int, plain_inventory: Mapping[int, Mapping[str, Any]] | None
    ) -> int | None:
        if ypos >= self.auto_pickaxe_y_threshold or not plain_inventory:
            return None
        for axe in self.pickaxe_priority:
            for slot, item in plain_inventory.items():
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == axe and 0 <= int(slot) <= 8:
                    return int(slot)
        return None


# ----------------------------------------------------------------------
# Y-dwell tracker (passive observation only - no environment intervention)
# ----------------------------------------------------------------------


class YDwellTrackerMixin:
    """Maintain a per-Y dwell counter that the Monitor module can read.

    NOT an intervention: this mixin never modifies the environment. It only
    accumulates how many env steps the agent has spent at each Y level, so
    that Monitor.ProgressScore (Eq. 2) and Monitor.DetectStagnation
    (Eq. 3) can compute their inputs without re-walking the trajectory.
    """

    def _reset_y_dwell(self) -> None:
        self._y_dwell: Dict[int, int] = {}

    def _update_y_dwell(self, ypos: int) -> None:
        self._y_dwell[ypos] = self._y_dwell.get(ypos, 0) + 1

    @property
    def y_dwell(self) -> Mapping[int, int]:
        return self._y_dwell


# ----------------------------------------------------------------------
# The actual wrapper
# ----------------------------------------------------------------------


class MineEvolveEnvWrapper(
    gym.Wrapper, DynamicOreSpawnMixin, AutoPickaxeMixin, YDwellTrackerMixin
):
    """Wrap a MineRL env with MineEvolve-specific runtime behaviour."""

    def __init__(self, env: gym.Env, cfg, logger: logging.Logger) -> None:
        gym.Wrapper.__init__(self, env)
        self.cfg = cfg
        self.logger = logger
        self.allow_open_inventory = False
        self.allow_change_hotbar = False

        self.recorder_mod = RecorderMod(cfg=cfg.get("record", {}), logger=logger)
        self.status_mod = StatusMod(cfg=cfg, logger=logger)
        self.task_checker = TaskCheckerMod(cfg=cfg)

        self._reset_ore_state()
        self._reset_y_dwell()
        self._current_subgoal: Tuple[str, int] | None = None
        self._current_subgoal_done = False
        self._latest_info: Dict[str, Any] = {}
        self.cache_obs: Any = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> Any:
        self._reset_ore_state()
        self._reset_y_dwell()
        self._current_subgoal = None
        self._current_subgoal_done = False
        self._latest_info = {}
        self.cache_obs = None

        self.recorder_mod.reset()
        self.status_mod.reset()
        self.task_checker.reset()

        obs = self.env.reset()
        self.cache_obs = obs
        if isinstance(obs, Mapping):
            self.status_mod.step(obs, {})
            self._latest_info = self.status_mod.get_status()

        commands = list(self.cfg.get("commands", []) or [])
        for cmd in commands:
            try:
                self.execute_cmd(cmd)
            except Exception as exc:
                self.logger.warning("Command failed: %s (%s)", cmd, exc)
        return obs

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _sanitize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in list(action.items()):
            if isinstance(value, list):
                action[key] = np.asarray(value)
        if not self.allow_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i + 1}"] = np.array(0)
            action["use"] = np.array(0)
            action["inventory"] = np.array(0)
        if not self.allow_open_inventory:
            action["inventory"] = np.array(0)
        action["drop"] = np.array(0)
        if int(action.get("attack", 0)) > 0:
            for k in ("jump", "left", "right", "sneak", "sprint"):
                if k in action:
                    action[k] = np.array(0)
        return action

    def step(
        self,
        action: Dict[str, Any],
        subgoal: Tuple[str, int] | None = None,
    ):
        action = self._sanitize_action(dict(action))

        if not self.allow_change_hotbar:
            ypos = self.status_mod.get_height()
            slot = self._select_best_pickaxe_slot(
                ypos=ypos, plain_inventory=self.status_mod.plain_inventory()
            )
            if slot is not None:
                action[f"hotbar.{slot + 1}"] = np.array(1)

        obs, reward, done, info = self.env.step(action)

        if subgoal is not None and (
            self._current_subgoal is None or subgoal[0] != self._current_subgoal[0]
        ):
            self.task_checker.reset(obs.get("inventory", {}))
            self._current_subgoal = subgoal

        self.recorder_mod.step(obs, action)
        self.status_mod.step(obs, action)

        info = dict(info)
        info.update(self.status_mod.get_status())
        info["isGuiOpen"] = bool(obs.get("isGuiOpen", False))

        ypos = self.status_mod.get_height()
        self._update_y_dwell(ypos)
        info["y_dwell_at_current_y"] = self._y_dwell.get(ypos, 0)
        self._maybe_spawn_ore(self.env, ypos)

        try:
            self._current_subgoal_done = self.task_checker.step(
                inventory=obs.get("inventory", {}),
                subgoal=subgoal,
            )
        except Exception as exc:
            self.logger.error("task checker failed: %s", exc)
            self._current_subgoal_done = True

        if self._current_subgoal_done:
            self._current_subgoal = None

        self._latest_info = info
        self.cache_obs = obs
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def current_subgoal_done(self) -> bool:
        return self._current_subgoal_done

    @property
    def info(self) -> Dict[str, Any]:
        return self._latest_info

    def execute_cmd(self, cmd: str) -> None:
        try:
            execute = getattr(self.env, "execute_cmd", None)
            if callable(execute):
                execute(cmd)
                return

            action = self.env.action_space.noop()
            action["chat"] = str(cmd)
            obs, _reward, _done, info = self.env.step(action)
            if isinstance(obs, Mapping):
                self.status_mod.step(obs, action)
                info = dict(info)
                info.update(self.status_mod.get_status())
                self._latest_info = info
                self.cache_obs = obs
        except Exception as exc:
            self.logger.warning("execute_cmd failed: %s (%s)", cmd, exc)

    def save_video(self, task: str, status: str) -> Any:
        return self.recorder_mod.save(task, status)
