"""RecorderMod: optional per-step POV/action recorder for offline inspection.

Saves frames + actions to disk only when ``cfg.record.video.save`` is True.
Implementation is intentionally simple - we use OpenCV's VideoWriter and a
JSONL action log, with no external dependencies beyond what's already
imported by the wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class RecorderMod:
    """Buffer POV frames + actions in memory; flush to disk on save()."""

    def __init__(self, cfg: Mapping[str, Any], logger: logging.Logger | None = None) -> None:
        cfg = dict(cfg or {})
        self.video_cfg = dict(cfg.get("video") or {})
        self.action_cfg = dict(cfg.get("action") or {})
        self.logger = logger or logging.getLogger("mineevolve.env.recorder")
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._actions: list[dict] = []

    def step(self, obs: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        if self.video_cfg.get("save", False):
            pov = obs.get("pov")
            if isinstance(pov, np.ndarray):
                self._frames.append(pov.copy())
        if self.action_cfg.get("save", False):
            self._actions.append({
                k: self._jsonable_action_value(v)
                for k, v in action.items()
                if not k.startswith("hotbar.") or self._action_value_nonzero(v)
            })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, task: str, status: str) -> threading.Thread | None:
        if not (self.video_cfg.get("save", False) or self.action_cfg.get("save", False)):
            return None

        frames = list(self._frames)
        actions = list(self._actions)
        video_path = self.video_cfg.get("path", "videos")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(task))[:80]
        out_dir = Path(video_path) / safe_task
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{timestamp}_{status}"

        def _flush() -> None:
            try:
                if frames:
                    self._write_video(out_dir / f"{stem}.mp4", frames)
                if actions:
                    with open(out_dir / f"{stem}.actions.jsonl", "w", encoding="utf-8") as fh:
                        for a in actions:
                            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
            except Exception as exc:
                self.logger.warning("recorder save failed: %s", exc)

        thread = threading.Thread(target=_flush, daemon=True)
        thread.start()
        self.reset()
        return thread

    @staticmethod
    def _write_video(path: Path, frames: list[np.ndarray]) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            return
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.fspath(path), fourcc, 20.0, (w, h))
        try:
            for f in frames:
                bgr = f[:, :, ::-1] if f.ndim == 3 else f
                writer.write(bgr)
        finally:
            writer.release()

    @staticmethod
    def _jsonable_action_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.item() if value.shape == () else value.tolist()
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted
        if hasattr(value, "__int__"):
            return int(value)
        return value

    @staticmethod
    def _action_value_nonzero(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(np.any(value != 0))
        try:
            return bool(value != 0)
        except Exception:
            return True
