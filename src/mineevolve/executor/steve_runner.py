"""Runtime adapter that drives the loaded STEVE-1 policy step-by-step.

We support two policy interfaces (from MineStudio or the native steve1
package) without copying any of their code. The adapter exposes a single
public method, ``act(obs)``, returning an action dict the wrapper can
``step()`` with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger("mineevolve.executor.steve_runner")


@dataclass
class SteveCondition:
    text: str
    cond_scale: float = 4.0


class SteveRunner:
    """Step-wise driver around a loaded STEVE-1 policy."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self._condition_obj: Any | None = None
        self._state: Any | None = None

    # ------------------------------------------------------------------
    # Condition lifecycle
    # ------------------------------------------------------------------

    def set_condition(self, condition: SteveCondition | str) -> None:
        if isinstance(condition, str):
            condition = SteveCondition(text=condition)

        prepare = getattr(self.policy, "prepare_condition", None)
        if callable(prepare):
            self._condition_obj = prepare({
                "cond_scale": float(condition.cond_scale),
                "text": str(condition.text),
            })
            initial_state = getattr(self.policy, "initial_state", None)
            if callable(initial_state):
                try:
                    self._state = initial_state(self._condition_obj, batch_size=1)
                except TypeError:
                    self._state = initial_state(condition=self._condition_obj)
            return

        # Native steve1 package path: encoders live on the agent itself.
        if hasattr(self.policy, "set_text_condition"):
            self.policy.set_text_condition(text=str(condition.text), scale=float(condition.cond_scale))
            self._condition_obj = condition
            self._state = None
            return

        # Last-resort: stash and apply per-step.
        self._condition_obj = condition
        initial_state = getattr(self.policy, "initial_state", None)
        if callable(initial_state):
            try:
                self._state = initial_state(condition=condition, batch_size=1)
            except TypeError:
                try:
                    self._state = initial_state(batch_size=1)
                except TypeError:
                    self._state = initial_state()
        else:
            self._state = None

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._condition_obj is None:
            raise RuntimeError("STEVE-1 condition not set. Call set_condition() first.")

        step_fn = getattr(self.policy, "get_steve_action", None)
        if callable(step_fn):
            action, self._state = step_fn(
                self._condition_obj,
                obs,
                self._state,
                input_shape="*",
            )
            return action

        action_fn = getattr(self.policy, "get_action", None)
        if callable(action_fn):
            try:
                return action_fn(obs)
            except TypeError as exc:
                try:
                    result = action_fn(obs, self._state)
                except TypeError:
                    raise exc
                if isinstance(result, tuple) and len(result) == 2:
                    action, self._state = result
                    return action
                return result

        raise RuntimeError("Loaded STEVE-1 policy exposes no recognised action method.")

    def reset_episode(self) -> None:
        self._condition_obj = None
        self._state = None
        reset = getattr(self.policy, "reset_episode", None)
        if callable(reset):
            reset()
