"""Hydra entry-point: client-side Algorithm 1 main loop.

Run with::

    python -m mineevolve.main benchmark=iron llm=qwen_plus

The server (FastAPI loading STEVE-1 + planner + 4 modules) must already be
running, e.g. via ``scripts/server.sh``. This client process owns the
MineRL environment and orchestrates the per-episode lifecycle in
Algorithm 1 of the paper.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .client import MineEvolveClient
from .monitors import StepMonitor, SuccessMonitor
from .util.logger import info_panel, print_results, setup_logging


logger = logging.getLogger("mineevolve.main")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get_evaluate_tasks(cfg: DictConfig) -> List[Tuple[int, str, str]]:
    """Resolve a benchmark yaml's ``evaluate`` list into [(id, type, instr)]."""

    all_tasks = list(cfg.benchmark.all_task)
    id_map = {int(t["id"]): t for t in all_tasks}
    selected = list(cfg.benchmark.get("evaluate") or [])
    if not selected:
        return [(int(t["id"]), str(t["type"]), str(t["instruction"])) for t in all_tasks]
    return [
        (int(i), str(id_map[int(i)]["type"]), str(id_map[int(i)]["instruction"]))
        for i in selected
        if int(i) in id_map
    ]


def _state_snapshot(env_info: Mapping[str, Any], task_goal: str) -> Dict[str, Any]:
    return {
        "task_goal": task_goal,
        "inventory": dict(env_info.get("inventory") or {}),
        "coords": list(env_info.get("coords") or [0, 64, 0]),
        "ypos": int(env_info.get("ypos") or 64),
        "health": float(env_info.get("health") or 20.0),
        "hunger": float(env_info.get("hunger") or 20.0),
        "isGuiOpen": bool(env_info.get("isGuiOpen") or False),
    }


# ----------------------------------------------------------------------
# Subgoal execution loop
# ----------------------------------------------------------------------


def _run_subgoal(
    env,
    client: MineEvolveClient,
    subgoal: Mapping[str, Any],
    task_goal: str,
    max_steps: int,
    coord_window_size: int = 256,
) -> Dict[str, Any]:
    """Execute one subgoal under the STEVE-1 policy until success / timeout."""

    target = None
    for c in subgoal.get("checks") or []:
        if isinstance(c, Mapping) and c.get("type") == "inv_ge":
            target = (str(c.get("item") or ""), int(c.get("n") or 1))
            break
    if target is None:
        target = (str(subgoal.get("condition") or ""), 1)

    coord_history: deque = deque(maxlen=coord_window_size)
    obs = env.cache_obs if hasattr(env, "cache_obs") else None
    if obs is None:
        try:
            obs = env.reset()
        except Exception:
            obs = {}

    start_inv = dict((env.info or {}).get("inventory") or {})
    start_coords = list((env.info or {}).get("coords") or [0, 64, 0])
    start_state = _state_snapshot(env.info or {}, task_goal)

    timeout_s = int(subgoal.get("timeout_s") or 60)
    deadline = time.monotonic() + max(1, timeout_s)

    success = False
    timed_out = False
    died = False
    steps = 0

    while steps < max_steps:
        if time.monotonic() > deadline:
            timed_out = True
            break

        # Ask server for next STEVE-1 action conditioned on the subgoal text.
        try:
            r = client.action(
                condition=str(subgoal.get("condition") or task_goal),
                obs={"image": _safe_pov(obs)},
            )
        except Exception as exc:
            logger.warning("server.action failed: %s", exc)
            break

        action = r.get("action")
        if action is None:
            logger.warning("server returned empty action: %s", r.get("error"))
            break

        try:
            obs, _reward, done, info = env.step(
                action,
                subgoal=target,
            )
        except Exception as exc:
            logger.exception("env.step crashed: %s", exc)
            break

        steps += 1
        coords = list(info.get("coords") or [0, 64, 0])
        coord_history.append(coords)

        if env.current_subgoal_done:
            success = True
            break
        if done:
            died = True
            break

    end_info = env.info or {}
    end_inv = dict(end_info.get("inventory") or {})
    end_coords = list(end_info.get("coords") or [0, 64, 0])

    delta_v = _diff_int_dict(start_inv, end_inv)
    delta_s = {
        "coords_start": start_coords,
        "coords_end": end_coords,
    }
    end_state = _state_snapshot(end_info, task_goal)

    return {
        "success": success,
        "timed_out": timed_out,
        "died": died,
        "steps": steps,
        "delta_v": delta_v,
        "delta_s": delta_s,
        "coords_history": list(coord_history),
        "end_state": end_state,
        "target_item": target[0] if target else None,
        "subgoal_id": str(subgoal.get("subgoal_id") or ""),
        "condition": str(subgoal.get("condition") or ""),
    }


def _safe_pov(obs: Any) -> Any:
    if isinstance(obs, Mapping):
        pov = obs.get("pov")
        if isinstance(pov, np.ndarray):
            return pov.tolist()
    return None


def _diff_int_dict(prev: Mapping[str, int], curr: Mapping[str, int]) -> Dict[str, int]:
    keys = set(prev) | set(curr)
    out: Dict[str, int] = {}
    for k in keys:
        delta = int(curr.get(k, 0)) - int(prev.get(k, 0))
        if delta != 0:
            out[str(k)] = delta
    return out


# ----------------------------------------------------------------------
# Episode loop (Algorithm 1)
# ----------------------------------------------------------------------


def run_episode(
    env,
    client: MineEvolveClient,
    task_goal: str,
    max_subgoals: int = 12,
    max_steps_per_subgoal: int = 1200,
    eta_fail: float = 0.5,
    recent_window: int = 4,
    budget_tokens: int = 512,
    top_k: int = 16,
) -> Tuple[bool, int]:
    """Algorithm 1: planner -> executor -> Monitor -> Inducer -> Curator -> Adaptor."""

    obs = env.reset()
    env_info = env.info or {}
    state = _state_snapshot(env_info, task_goal)

    client.reset(task_goal=task_goal)
    plan_resp = client.plan(state=state, budget_tokens=budget_tokens, top_k=top_k)
    plan = plan_resp.get("plan") or {}
    subgoals: List[Dict[str, Any]] = list(plan.get("subgoals") or [])

    if not subgoals:
        logger.warning("Planner returned no subgoals for task %s", task_goal)
        return False, 0

    consecutive_failures = 0
    total_steps = 0
    i = 0
    while i < min(len(subgoals), max_subgoals):
        sg = subgoals[i]
        result = _run_subgoal(
            env=env,
            client=client,
            subgoal=sg,
            task_goal=task_goal,
            max_steps=max_steps_per_subgoal,
        )
        total_steps += int(result["steps"])

        # Push typed feedback to server (Monitor stage 1)
        try:
            client.monitor(
                subgoal=[result["condition"], 1],
                target_item=result["target_item"],
                coords_history=result["coords_history"],
                delta_v=result["delta_v"],
                delta_s=result["delta_s"],
                gui_open=bool(result["end_state"].get("isGuiOpen", False)),
                success=int(bool(result["success"])),
                timed_out=bool(result["timed_out"]),
                died=bool(result["died"]),
                p_goal=0.0,
                subgoal_id=result["subgoal_id"],
                coords_now=result["end_state"].get("coords", []),
            )
        except Exception as exc:
            logger.warning("client.monitor failed: %s", exc)

        # Inducer + Curator (stage 2/3 of Algorithm 1, plus Figure 3 deadlock)
        try:
            client.induce(
                eta_fail=eta_fail,
                recent_window=recent_window,
                state=result["end_state"],
            )
        except Exception as exc:
            logger.warning("client.induce failed: %s", exc)

        if result["success"]:
            consecutive_failures = 0
            client.advance(success=True)
            i += 1
            continue

        # Failure or stagnation: ask Adaptor to repair the suffix
        consecutive_failures += 1
        try:
            repair_resp = client.repair(
                state=result["end_state"],
                recent_window=recent_window,
                eta_fail=eta_fail,
                budget_tokens=budget_tokens,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("client.repair failed: %s", exc)
            repair_resp = {"repaired": False}

        if repair_resp.get("repaired"):
            new_plan = repair_resp.get("plan") or {}
            new_subgoals = list(new_plan.get("subgoals") or [])
            if new_subgoals:
                subgoals = new_subgoals
                continue  # do not advance i; retry from the same logical position

        # No repair; advance to next subgoal to avoid infinite retry on the same step
        client.advance(success=False)
        i += 1
        if consecutive_failures >= 3:
            logger.warning("Aborting episode after %d consecutive failures", consecutive_failures)
            break

    end_inv = dict((env.info or {}).get("inventory") or {})
    success = _episode_succeeded(task_goal=task_goal, inventory=end_inv)
    return success, total_steps


def _episode_succeeded(task_goal: str, inventory: Mapping[str, int]) -> bool:
    """Heuristic: success if the noun phrase in task_goal appears in inventory."""

    goal = str(task_goal).lower()
    for k in inventory:
        token = str(k).lower().replace("_", " ")
        if token in goal and int(inventory.get(k, 0)) > 0:
            return True
    return False


# ----------------------------------------------------------------------
# Hydra entry
# ----------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    setup_logging()
    info_panel(f"MineEvolve: {cfg.benchmark.env.name}  |  llm={cfg.llm.provider}/{cfg.llm.model}")

    # Lazy imports so the module can be examined without a MineRL install
    from .env import make_env

    client = MineEvolveClient(
        base_url=f"{cfg.server.url}:{cfg.server.port}",
        timeout=float(cfg.server.timeout),
    )

    env = make_env(cfg, logger=logger)

    tasks = _get_evaluate_tasks(cfg)
    success_mon = SuccessMonitor()
    step_mon = StepMonitor()
    times = int(cfg.benchmark.env.get("times") or 1)

    for task_id, task_type, instruction in tasks:
        for run_idx in range(times):
            info_panel(f"task {task_id} ({task_type})  run {run_idx + 1}/{times}: {instruction}")
            try:
                success, steps = run_episode(
                    env=env,
                    client=client,
                    task_goal=instruction,
                    max_subgoals=int(cfg.runtime.get("max_subgoals", 12)),
                    max_steps_per_subgoal=int(cfg.runtime.get("max_steps_per_subgoal", 1200)),
                    eta_fail=float(cfg.runtime.get("eta_fail", 0.5)),
                    recent_window=int(cfg.runtime.get("recent_window", 4)),
                    budget_tokens=int(cfg.runtime.get("budget_tokens", 512)),
                    top_k=int(cfg.runtime.get("top_k", 16)),
                )
            except Exception as exc:
                logger.exception("run_episode failed: %s", exc)
                success, steps = False, 0
            success_mon.record(instruction, success)
            step_mon.record(instruction, steps)

    print_results(
        title=f"Results: {cfg.benchmark.env.name}",
        tasks=[t[2] for t in tasks],
        success=success_mon,
        step=step_mon,
    )

    OmegaConf.save(config=cfg, f="resolved_config.yaml")


if __name__ == "__main__":
    main()
