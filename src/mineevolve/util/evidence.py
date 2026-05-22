"""Detailed per-subgoal evidence capture for induction.

The Inducer should not infer skills/remedies from a single success bit.  This
module writes a full step trace to disk and returns a compact, prompt-friendly
summary that is embedded into TypedFeedback via ``delta_s["evidence"]``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


IMPORTANT_ACTION_KEYS: tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
    "camera",
    "inventory",
    "drop",
    "chat",
)


@dataclass
class StepEvidence:
    step: int
    t: float
    action: Dict[str, Any]
    coords: list[float]
    inventory: Dict[str, int]
    inventory_diff: Dict[str, int]
    gui_open: bool
    health: float | None = None
    hunger: float | None = None
    reward: float | None = None
    done: bool = False
    current_subgoal_done: bool = False
    frame_path: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "t": round(self.t, 3),
            "action": self.action,
            "coords": self.coords,
            "inventory": self.inventory,
            "inventory_diff": self.inventory_diff,
            "gui_open": self.gui_open,
            "health": self.health,
            "hunger": self.hunger,
            "reward": self.reward,
            "done": self.done,
            "current_subgoal_done": self.current_subgoal_done,
            "frame_path": self.frame_path,
        }


@dataclass
class SubgoalEvidenceRecorder:
    """Capture dense trace files and a compact LLM evidence summary."""

    root: Path
    task_goal: str
    subgoal_id: str
    condition: str
    target_item: str | None
    keyframe_interval: int = 40
    max_prompt_events: int = 24
    _start_time: float = field(default_factory=time.monotonic)
    _steps: list[StepEvidence] = field(default_factory=list)
    _keyframes: list[dict] = field(default_factory=list)
    _events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def record_step(
        self,
        step: int,
        action: Mapping[str, Any],
        obs: Mapping[str, Any] | None,
        info: Mapping[str, Any],
        reward: float | None,
        done: bool,
        current_subgoal_done: bool,
    ) -> None:
        inv = _normalize_inventory(info.get("inventory", {}))
        inv_diff = _normalize_inventory_delta(info.get("inventory_diff", {}))
        coords = _coords(info.get("coords"))
        should_frame = (
            step == 1
            or step % max(1, int(self.keyframe_interval)) == 0
            or bool(inv_diff)
            or bool(current_subgoal_done)
            or bool(done)
        )
        frame_path = None
        if should_frame and isinstance(obs, Mapping):
            frame_path = self._save_frame(step, obs.get("pov"))

        ev = StepEvidence(
            step=step,
            t=time.monotonic() - self._start_time,
            action=summarize_action(action),
            coords=coords,
            inventory=inv,
            inventory_diff=inv_diff,
            gui_open=bool(info.get("isGuiOpen", False)),
            health=_maybe_float(info.get("health")),
            hunger=_maybe_float(info.get("hunger")),
            reward=_maybe_float(reward),
            done=bool(done),
            current_subgoal_done=bool(current_subgoal_done),
            frame_path=frame_path,
        )
        self._steps.append(ev)

        if frame_path:
            self._keyframes.append(
                {
                    "step": step,
                    "path": frame_path,
                    "coords": coords,
                    "inventory": inv,
                    "inventory_diff": inv_diff,
                }
            )
        if inv_diff or current_subgoal_done or done or ev.gui_open:
            self._events.append(
                {
                    "step": step,
                    "event": _event_label(inv_diff, current_subgoal_done, done, ev.gui_open),
                    "coords": coords,
                    "inventory_diff": inv_diff,
                    "inventory": inv,
                    "action": ev.action,
                    "frame_path": frame_path,
                }
            )

    def finalize(
        self,
        success: bool,
        timed_out: bool,
        died: bool,
        start_state: Mapping[str, Any],
        end_state: Mapping[str, Any],
        delta_v: Mapping[str, int],
        delta_s: Mapping[str, Any],
    ) -> Dict[str, Any]:
        trace_path = self.root / "steps.jsonl"
        summary_path = self.root / "summary.json"
        with open(trace_path, "w", encoding="utf-8") as fh:
            for ev in self._steps:
                fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

        coord_path = [ev.coords for ev in self._steps if ev.coords]
        summary = {
            "task_goal": self.task_goal,
            "subgoal_id": self.subgoal_id,
            "condition": self.condition,
            "target_item": self.target_item,
            "success": bool(success),
            "timed_out": bool(timed_out),
            "died": bool(died),
            "steps": len(self._steps),
            "duration_s": round(time.monotonic() - self._start_time, 3),
            "start_state": _compact_state(start_state),
            "end_state": _compact_state(end_state),
            "delta_v": dict(delta_v),
            "delta_s": dict(delta_s),
            "coord_path_sample": _sample(coord_path, limit=12),
            "action_histogram": _action_histogram(self._steps),
            "event_timeline": self._events[: self.max_prompt_events],
            "keyframes": self._keyframes[: self.max_prompt_events],
            "trace_jsonl": str(trace_path),
            "summary_json": str(summary_path),
            "evidence_note": (
                "Keyframes are local PNG files captured from the POV observation. "
                "The LLM prompt receives their paths plus state/action metadata; "
                "the JSONL file contains the dense per-step trace."
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return summary

    def _save_frame(self, step: int, pov: Any) -> str | None:
        if not isinstance(pov, np.ndarray) or pov.ndim < 2:
            return None
        path = self.frames_dir / f"step_{step:05d}.png"
        try:
            from PIL import Image

            arr = pov
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            return str(path)
        except Exception:
            return None


def make_subgoal_evidence_dir(
    artifact_dir: str | None,
    task_id: int | None,
    run_idx: int,
    subgoal_id: str,
    condition: str,
) -> Path | None:
    if not artifact_dir:
        return None
    safe_condition = "".join(c if c.isalnum() or c in "_-" else "_" for c in condition)[:48]
    task_part = f"task_{task_id}" if task_id is not None else "task_unknown"
    return (
        Path(artifact_dir)
        / "evidence"
        / task_part
        / f"run_{run_idx + 1}"
        / f"{subgoal_id or 'sg_unknown'}_{safe_condition}"
    )


def summarize_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in IMPORTANT_ACTION_KEYS:
        if key not in action:
            continue
        value = _jsonable(action[key])
        if _nonzero(value) or key in {"camera", "chat"}:
            out[key] = value
    for i in range(9):
        key = f"hotbar.{i + 1}"
        if key in action and _nonzero(_jsonable(action[key])):
            out[key] = _jsonable(action[key])
    return out


def _sample(values: Sequence[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return list(values)
    if limit <= 2:
        return [values[0], values[-1]]
    stride = max(1, (len(values) - 2) // (limit - 2))
    sampled = [values[0]]
    sampled.extend(values[i] for i in range(stride, len(values) - 1, stride)[: limit - 2])
    sampled.append(values[-1])
    return sampled[:limit]


def _action_histogram(steps: Sequence[StepEvidence]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ev in steps:
        for key, value in ev.action.items():
            if key == "camera":
                continue
            if _nonzero(value):
                hist[key] = hist.get(key, 0) + 1
    return hist


def _event_label(inv_diff: Mapping[str, int], subgoal_done: bool, done: bool, gui_open: bool) -> str:
    labels: list[str] = []
    if inv_diff:
        labels.append("inventory_changed")
    if subgoal_done:
        labels.append("subgoal_completed")
    if done:
        labels.append("episode_done")
    if gui_open:
        labels.append("gui_open")
    return "+".join(labels) or "state_event"


def _compact_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "inventory": dict(state.get("inventory") or {}),
        "coords": _coords(state.get("coords")),
        "ypos": state.get("ypos"),
        "health": state.get("health"),
        "hunger": state.get("hunger"),
        "isGuiOpen": bool(state.get("isGuiOpen", False)),
    }


def _coords(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[float] = []
    for x in list(value)[:3]:
        try:
            out.append(round(float(x), 3))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _normalize_inventory(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            count = int(v)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[str(k)] = count
    return out


def _normalize_inventory_delta(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            delta = int(v)
        except (TypeError, ValueError):
            continue
        if delta != 0:
            out[str(k)] = delta
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _nonzero(value: Any) -> bool:
    if isinstance(value, list):
        return any(_nonzero(v) for v in value)
    if isinstance(value, Mapping):
        return any(_nonzero(v) for v in value.values())
    try:
        return bool(value != 0)
    except Exception:
        return bool(value)


def _maybe_float(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
