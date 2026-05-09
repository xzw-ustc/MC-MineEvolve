"""HTTP client used by the env-side process to talk to the FastAPI server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger("mineevolve.client")


class MineEvolveClient:
    """Thin requests wrapper for /reset and /chat."""

    def __init__(self, base_url: str = "http://127.0.0.1:9000", timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _post(self, path: str, json: Mapping[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self._session.post(url, json=json, timeout=self.timeout)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def _chat(self, type_: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/chat", {"type": type_, "payload": dict(payload)})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_goal: str) -> Dict[str, Any]:
        return self._post("/reset", {"task_goal": task_goal})

    def status(self) -> Dict[str, Any]:
        return self._chat("status", {})

    def plan(
        self,
        state: Mapping[str, Any],
        budget_tokens: int = 512,
        top_k: int = 16,
    ) -> Dict[str, Any]:
        return self._chat(
            "plan",
            {"state": dict(state), "budget_tokens": int(budget_tokens), "top_k": int(top_k)},
        )

    def action(self, condition: str, obs: Mapping[str, Any]) -> Dict[str, Any]:
        return self._chat("action", {"condition": str(condition), "obs": dict(obs)})

    def monitor(
        self,
        subgoal: Sequence[Any],
        target_item: str | None,
        coords_history: Sequence[Sequence[float]],
        delta_v: Mapping[str, int],
        delta_s: Mapping[str, Any],
        gui_open: bool,
        success: int,
        timed_out: bool = False,
        died: bool = False,
        p_goal: float = 0.0,
        subgoal_id: str | None = None,
        coords_now: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        return self._chat(
            "monitor",
            {
                "subgoal": list(subgoal),
                "target_item": target_item,
                "coords_history": [list(c) for c in coords_history],
                "delta_v": dict(delta_v),
                "delta_s": dict(delta_s),
                "gui_open": bool(gui_open),
                "success": int(success),
                "timed_out": bool(timed_out),
                "died": bool(died),
                "p_goal": float(p_goal),
                "subgoal_id": subgoal_id,
                "coords_now": list(coords_now) if coords_now is not None else None,
            },
        )

    def induce(
        self,
        eta_fail: float = 0.5,
        recent_window: int = 4,
        state: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._chat(
            "induce",
            {
                "eta_fail": float(eta_fail),
                "recent_window": int(recent_window),
                "state": dict(state or {}),
            },
        )

    def repair(
        self,
        state: Mapping[str, Any],
        recent_window: int = 4,
        eta_fail: float = 0.5,
        budget_tokens: int = 512,
        top_k: int = 16,
    ) -> Dict[str, Any]:
        return self._chat(
            "repair",
            {
                "state": dict(state),
                "recent_window": int(recent_window),
                "eta_fail": float(eta_fail),
                "budget_tokens": int(budget_tokens),
                "top_k": int(top_k),
            },
        )

    def advance(self, success: bool) -> Dict[str, Any]:
        return self._chat("advance", {"success": bool(success)})
