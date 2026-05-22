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
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .client import MineEvolveClient
from .executor import CraftHelper, CraftRequest
from .monitors import StepMonitor, SuccessMonitor
from .util.evidence import SubgoalEvidenceRecorder, make_subgoal_evidence_dir
from .util.items import inventory_satisfies
from .util.logger import info_panel, print_results, setup_logging


logger = logging.getLogger("mineevolve.main")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get_evaluate_tasks(cfg: DictConfig) -> List[Tuple[int, str, str]]:
    """Resolve a benchmark yaml's ``evaluate`` list into [(id, type, instr)]."""

    benchmark_cfg = _benchmark_cfg(cfg)
    all_tasks = list(benchmark_cfg.all_task)
    id_map = {int(t["id"]): t for t in all_tasks}
    selected = list(benchmark_cfg.get("evaluate") or [])
    if not selected:
        return [(int(t["id"]), str(t["type"]), str(t["instruction"])) for t in all_tasks]
    return [
        (int(i), str(id_map[int(i)]["type"]), str(id_map[int(i)]["instruction"]))
        for i in selected
        if int(i) in id_map
    ]


def _benchmark_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.benchmark if "benchmark" in cfg else cfg


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
    default_timeout_s: int = 60,
    coord_window_size: int = 256,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
    evidence_keyframe_interval: int = 40,
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
    evidence_dir = make_subgoal_evidence_dir(
        artifact_dir=artifact_dir,
        task_id=task_id,
        run_idx=run_idx,
        subgoal_id=str(subgoal.get("subgoal_id") or ""),
        condition=str(subgoal.get("condition") or ""),
    )
    evidence = (
        SubgoalEvidenceRecorder(
            root=evidence_dir,
            task_goal=task_goal,
            subgoal_id=str(subgoal.get("subgoal_id") or ""),
            condition=str(subgoal.get("condition") or ""),
            target_item=target[0] if target else None,
            keyframe_interval=evidence_keyframe_interval,
        )
        if evidence_dir is not None
        else None
    )

    timeout_s = max(int(subgoal.get("timeout_s") or 0), int(default_timeout_s))
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
                obs={"image": _safe_pov(obs), "pov": _safe_pov(obs)},
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
        if evidence is not None:
            evidence.record_step(
                step=steps,
                action=action,
                obs=obs if isinstance(obs, Mapping) else {},
                info=info,
                reward=_reward,
                done=done,
                current_subgoal_done=bool(env.current_subgoal_done),
            )

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
    if evidence is not None:
        delta_s["evidence"] = evidence.finalize(
            success=success,
            timed_out=timed_out,
            died=died,
            start_state=start_state,
            end_state=end_state,
            delta_v=delta_v,
            delta_s={k: v for k, v in delta_s.items() if k != "evidence"},
        )

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


def _run_helper_subgoal(
    env,
    subgoal: Mapping[str, Any],
    task_goal: str,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
) -> Dict[str, Any]:
    """Execute one non-STEVE helper subgoal such as mc_craft or mc_smelt."""

    target = None
    for c in subgoal.get("checks") or []:
        if isinstance(c, Mapping) and c.get("type") == "inv_ge":
            target = (str(c.get("item") or ""), int(c.get("n") or 1))
            break
    if target is None:
        target = (str(subgoal.get("condition") or ""), 1)

    start_inv = dict((env.info or {}).get("inventory") or {})
    start_coords = list((env.info or {}).get("coords") or [0, 64, 0])
    start_state = _state_snapshot(env.info or {}, task_goal)

    hint = str(subgoal.get("executor_hint") or "").strip().lower()
    success = CraftHelper(env).execute(
        CraftRequest(
            kind=hint,
            target=target[0],
            quantity=target[1],
            timeout_s=float(subgoal.get("timeout_s") or 10),
        )
    )

    end_info = env.info or {}
    end_inv = dict(end_info.get("inventory") or {})
    end_coords = list(end_info.get("coords") or start_coords)
    delta_v = _diff_int_dict(start_inv, end_inv)
    delta_s = {
        "coords_start": start_coords,
        "coords_end": end_coords,
        "state_start": start_state,
    }
    end_state = _state_snapshot(end_info, task_goal)
    evidence_dir = make_subgoal_evidence_dir(
        artifact_dir=artifact_dir,
        task_id=task_id,
        run_idx=run_idx,
        subgoal_id=str(subgoal.get("subgoal_id") or ""),
        condition=str(subgoal.get("condition") or ""),
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "task_goal": task_goal,
            "subgoal_id": str(subgoal.get("subgoal_id") or ""),
            "condition": str(subgoal.get("condition") or ""),
            "target_item": target[0] if target else None,
            "success": bool(success),
            "timed_out": False,
            "died": False,
            "steps": 0,
            "start_state": start_state,
            "end_state": end_state,
            "delta_v": delta_v,
            "delta_s": {k: v for k, v in delta_s.items() if k != "evidence"},
            "event_timeline": [
                {
                    "step": 0,
                    "event": "helper_executed",
                    "executor_hint": hint,
                    "inventory_diff": delta_v,
                    "inventory": end_inv,
                    "coords": end_coords,
                }
            ],
        }
        summary_path = evidence_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        summary["summary_json"] = str(summary_path)
        delta_s["evidence"] = summary

    return {
        "success": bool(success),
        "timed_out": False,
        "died": False,
        "steps": 0,
        "delta_v": delta_v,
        "delta_s": delta_s,
        "coords_history": [start_coords, end_coords],
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
    subgoal_timeout_s: int = 60,
    eta_fail: float = 0.5,
    recent_window: int = 4,
    budget_tokens: int = 512,
    top_k: int = 16,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
    evidence_keyframe_interval: int = 40,
) -> Tuple[bool, int]:
    """Algorithm 1: planner -> executor -> Monitor -> Inducer -> Curator -> Adaptor."""

    obs = env.reset()
    env_info = env.info or {}
    state = _state_snapshot(env_info, task_goal)

    client.reset(task_goal=task_goal)
    plan_resp = client.plan(state=state, budget_tokens=budget_tokens, top_k=top_k)
    plan = plan_resp.get("plan") or {}
    _save_plan_artifact(
        artifact_dir=artifact_dir,
        task_goal=task_goal,
        task_id=task_id,
        run_idx=run_idx,
        stage="initial",
        plan=plan,
        extra={
            "retrieved_skill_ids": plan_resp.get("retrieved_skill_ids", []),
            "retrieved_remedy_ids": plan_resp.get("retrieved_remedy_ids", []),
        },
    )
    subgoals: List[Dict[str, Any]] = list(plan.get("subgoals") or [])

    if not subgoals:
        logger.warning("Planner returned no subgoals for task %s", task_goal)
        return False, 0

    consecutive_failures = 0
    total_steps = 0
    i = 0
    while i < min(len(subgoals), max_subgoals):
        sg = subgoals[i]
        executor_hint = str(sg.get("executor_hint") or "stevei").strip().lower()
        if executor_hint in {"mc_craft", "mc_smelt", "place", "use"}:
            result = _run_helper_subgoal(
                env=env,
                subgoal=sg,
                task_goal=task_goal,
                artifact_dir=artifact_dir,
                task_id=task_id,
                run_idx=run_idx,
            )
        else:
            result = _run_subgoal(
                env=env,
                client=client,
                subgoal=sg,
                task_goal=task_goal,
                max_steps=max_steps_per_subgoal,
                default_timeout_s=subgoal_timeout_s,
                artifact_dir=artifact_dir,
                task_id=task_id,
                run_idx=run_idx,
                evidence_keyframe_interval=evidence_keyframe_interval,
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
            _save_plan_artifact(
                artifact_dir=artifact_dir,
                task_goal=task_goal,
                task_id=task_id,
                run_idx=run_idx,
                stage=f"repair_i{i}",
                plan=new_plan,
                extra={
                    "active_remedies_used": repair_resp.get("active_remedies_used", []),
                    "deadlock_signal": repair_resp.get("deadlock_signal"),
                },
            )
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


def _save_plan_artifact(
    artifact_dir: str | None,
    task_goal: str,
    task_id: int | None,
    run_idx: int,
    stage: str,
    plan: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    if not artifact_dir:
        return
    out_dir = Path(artifact_dir) / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_goal)[:60]
    task_part = f"task_{task_id}" if task_id is not None else "task_unknown"
    path = out_dir / f"{task_part}_run_{run_idx + 1}_{stage}_{safe_goal}.json"
    payload = {
        "task_id": task_id,
        "run_idx": run_idx,
        "task_goal": task_goal,
        "stage": stage,
        "subgoal_count": len(list(plan.get("subgoals") or [])) if isinstance(plan, Mapping) else 0,
        "plan": dict(plan or {}),
        "extra": dict(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _episode_succeeded(task_goal: str, inventory: Mapping[str, int]) -> bool:
    """Heuristic: success if the noun phrase in task_goal appears in inventory."""

    goal = str(task_goal).lower()
    if any(word in goal for word in ("log", "wood", "tree")) and any(
        inventory_satisfies(inventory, target, 1) for target in ("log", "oak_log", "wood")
    ):
        return True
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
    benchmark_cfg = _benchmark_cfg(cfg)
    info_panel(f"MineEvolve: {benchmark_cfg.env.name}  |  llm={cfg.llm.provider}/{cfg.llm.model}")

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
    times = int(benchmark_cfg.env.get("times") or 1)

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
                    subgoal_timeout_s=int(cfg.runtime.get("subgoal_timeout_s", 60)),
                    eta_fail=float(cfg.runtime.get("eta_fail", 0.5)),
                    recent_window=int(cfg.runtime.get("recent_window", 4)),
                    budget_tokens=int(cfg.runtime.get("budget_tokens", 512)),
                    top_k=int(cfg.runtime.get("top_k", 16)),
                    artifact_dir=str(OmegaConf.select(cfg, "artifact_dir") or "."),
                    task_id=task_id,
                    run_idx=run_idx,
                    evidence_keyframe_interval=int(
                        OmegaConf.select(cfg, "record.evidence.keyframe_interval") or 40
                    ),
                )
            except Exception as exc:
                logger.exception("run_episode failed: %s", exc)
                success, steps = False, 0
            success_mon.record(instruction, success)
            step_mon.record(instruction, steps)
            save_video = getattr(env, "save_video", None)
            if callable(save_video):
                status = "success" if success else "failed"
                thread = save_video(instruction, status)
                if thread is not None:
                    thread.join(timeout=30.0)

    print_results(
        title=f"Results: {benchmark_cfg.env.name}",
        tasks=[t[2] for t in tasks],
        success=success_mon,
        step=step_mon,
    )

    OmegaConf.save(config=cfg, f="resolved_config.yaml")


if __name__ == "__main__":
    main()
