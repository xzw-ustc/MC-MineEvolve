"""Agent state container that ties the four MineEvolve modules together.

Lives inside the FastAPI server process. Each ``/chat`` request type maps to
one method here; the client process holds no MineEvolve module state of its
own and only forwards observations + outcomes via HTTP.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Mapping, Sequence

from ..adaptor import (
    assemble_repaired_plan,
    need_repair,
    repair_plan,
)
from ..curator import Curator, KnowledgeStore
from ..executor import SteveCondition, SteveRunner, load_steve_policy
from ..inducer import (
    KnowledgeEntry,
    induce_remedy_if_needed,
    induce_skills,
)
from ..monitor import FeedbackBuffer, Monitor, ProgressWeights, TypedFeedback
from ..planner import (
    Plan,
    Subgoal,
    generate_initial_plan,
    get_backend,
    make_chat_callable,
)


class MineEvolveAgent:
    """Per-process agent that hosts planner / 4 modules / STEVE-1 driver."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger("mineevolve.server.agent")
        self._lock = threading.Lock()

        self.weights = ProgressWeights(**dict(cfg.get("monitor", {}) or {}))
        self.monitor = Monitor(weights=self.weights)
        self.buffer = FeedbackBuffer()

        kb_path = str(cfg.get("memory", {}).get("path") or "memories/run")
        self.store = KnowledgeStore(path=kb_path)
        self.curator = Curator(self.store)

        self.planner_backend = get_backend(dict(cfg.get("llm", {}) or {}))
        self.llm_chat = make_chat_callable(self.planner_backend)

        self._steve: SteveRunner | None = None
        if bool(cfg.get("load_steve", True)):
            try:
                policy = load_steve_policy(**dict(cfg.get("steve", {}) or {}))
                self._steve = SteveRunner(policy)
            except Exception as exc:
                self.logger.warning("STEVE-1 unavailable: %s", exc)
                self._steve = None

        self._task_goal: str = ""
        self._current_plan: Plan | None = None
        self._next_index: int = 0

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self, task_goal: str) -> Dict[str, Any]:
        with self._lock:
            self.buffer.reset()
            self._task_goal = str(task_goal)
            self._current_plan = None
            self._next_index = 0
            if self._steve is not None:
                self._steve.reset_episode()
            return {"ok": True, "task_goal": self._task_goal}

    # ------------------------------------------------------------------
    # /chat type=plan
    # ------------------------------------------------------------------

    def plan(
        self,
        state: Mapping[str, Any],
        budget_tokens: int = 512,
        top_k: int = 16,
    ) -> Dict[str, Any]:
        with self._lock:
            retrieved = self.curator.retrieve(
                task_goal=self._task_goal,
                current_subgoal="",
                inventory=state.get("inventory") if isinstance(state, Mapping) else None,
                recent_failure_types=[],
                budget_tokens=budget_tokens,
                top_k=top_k,
            )
            plan = generate_initial_plan(
                backend=self.planner_backend,
                task_goal=self._task_goal,
                state=state,
                completed_prefix=(),
                retrieved_skills=retrieved["skills"],
                active_remedies=retrieved["remedies"],
            )
            if plan is None:
                # Degenerate fallback: a single subgoal mirroring the task.
                plan = Plan(
                    plan_id=f"p_{uuid.uuid4().hex[:8]}",
                    subgoals=[
                        Subgoal(
                            subgoal_id="sg_001",
                            condition=self._task_goal,
                            task_kind="mine",
                            executor_hint="stevei",
                        )
                    ],
                )
            self._current_plan = plan
            self._next_index = 0
            return {
                "plan": plan.to_dict(),
                "retrieved_skill_ids": retrieved["skill_ids"],
                "retrieved_remedy_ids": retrieved["remedy_ids"],
            }

    # ------------------------------------------------------------------
    # /chat type=action  (STEVE-1 single-step)
    # ------------------------------------------------------------------

    def action(self, condition: str, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._steve is None:
            return {"action": None, "error": "STEVE-1 not loaded"}
        with self._lock:
            self._steve.set_condition(SteveCondition(text=str(condition)))
            try:
                action = self._steve.act(obs)
            except Exception as exc:
                return {"action": None, "error": str(exc)}
        return {"action": action}

    # ------------------------------------------------------------------
    # /chat type=monitor  (push raw subgoal trace -> typed feedback)
    # ------------------------------------------------------------------

    def monitor_subgoal(
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
        with self._lock:
            sg_tuple = (str(subgoal[0]) if isinstance(subgoal, (list, tuple)) and subgoal else str(subgoal),
                        int(subgoal[1]) if isinstance(subgoal, (list, tuple)) and len(subgoal) > 1 else 1)
            fb = self.monitor.build_feedback(
                subgoal=sg_tuple,
                target_item=target_item,
                coords_history=coords_history,
                delta_v=delta_v,
                delta_s=delta_s,
                gui_open=gui_open,
                success=success,
                timed_out=timed_out,
                died=died,
                p_goal=p_goal,
                subgoal_id=subgoal_id,
                coords_now=coords_now,
            )
            self.buffer.append(fb)
            return fb.to_dict()

    # ------------------------------------------------------------------
    # /chat type=induce  (turn buffer segments into knowledge)
    # ------------------------------------------------------------------

    def induce_and_curate(
        self,
        eta_fail: float = 0.5,
        recent_window: int = 4,
        deadlock_window: int = 8,
        deadlock_min_distinct_subgoals: int = 2,
        deadlock_min_repeats: int = 3,
        state: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        with self._lock:
            successful_segments = self.buffer.successful_segments(min_len=1)
            recent = self.buffer.recent(recent_window)
            current_subgoal = ""
            if (
                self._current_plan is not None
                and 0 <= self._next_index < len(self._current_plan.subgoals)
            ):
                current_subgoal = self._current_plan.subgoals[self._next_index].condition

            # Cross-subgoal deadlock detection (Figure 3 task-level trigger)
            deadlock_signal = self.buffer.detect_deadlock(
                window=deadlock_window,
                min_distinct_subgoals=deadlock_min_distinct_subgoals,
                min_repeats=deadlock_min_repeats,
            )

            candidates: List[KnowledgeEntry] = []
            candidates.extend(
                induce_skills(
                    task_goal=self._task_goal,
                    segments=successful_segments,
                    llm_chat=self.llm_chat,
                )
            )
            remedy = induce_remedy_if_needed(
                current_subgoal=current_subgoal,
                recent_segment=recent,
                llm_chat=self.llm_chat,
                eta_fail=eta_fail,
                deadlock_signal=deadlock_signal,
            )
            if remedy is not None:
                candidates.append(remedy)

            report = self.curator.ingest(
                candidates=candidates,
                state_fields=state or {},
            )
            return {
                "candidates_n": len(candidates),
                "accepted": report["accepted"],
                "rejected": report["rejected"],
                "deadlock_signal": deadlock_signal,
                "store_stats": self.store.stats(),
            }

    # ------------------------------------------------------------------
    # /chat type=repair  (Adaptor local replanning)
    # ------------------------------------------------------------------

    def repair(
        self,
        state: Mapping[str, Any],
        recent_window: int = 4,
        eta_fail: float = 0.5,
        budget_tokens: int = 512,
        top_k: int = 16,
        deadlock_window: int = 8,
        deadlock_min_distinct_subgoals: int = 2,
        deadlock_min_repeats: int = 3,
    ) -> Dict[str, Any]:
        with self._lock:
            recent = self.buffer.recent(recent_window)
            deadlock_signal = self.buffer.detect_deadlock(
                window=deadlock_window,
                min_distinct_subgoals=deadlock_min_distinct_subgoals,
                min_repeats=deadlock_min_repeats,
            )
            if not need_repair(recent, eta_fail=eta_fail) and not deadlock_signal:
                return {"repaired": False, "reason": "trigger_not_met"}
            if self._current_plan is None:
                return {"repaired": False, "reason": "no_active_plan"}

            recent_failure_types = [e.f for e in recent if e.f and e.f != "NONE"]
            if deadlock_signal:
                recent_failure_types.append(str(deadlock_signal["failure_type"]))
            current_subgoal = ""
            if 0 <= self._next_index < len(self._current_plan.subgoals):
                current_subgoal = self._current_plan.subgoals[self._next_index].condition

            retrieved = self.curator.retrieve(
                task_goal=self._task_goal,
                current_subgoal=current_subgoal,
                inventory=state.get("inventory") if isinstance(state, Mapping) else None,
                recent_failure_types=recent_failure_types,
                budget_tokens=budget_tokens,
                top_k=top_k,
            )
            skills = [KnowledgeEntry.from_dict(k) for k in retrieved["skills"]]
            remedies = [KnowledgeEntry.from_dict(k) for k in retrieved["remedies"]]

            # Algorithm 3 line 26: build temporary remedy from current failure
            tmp_remedy = induce_remedy_if_needed(
                current_subgoal=current_subgoal,
                recent_segment=recent,
                llm_chat=self.llm_chat,
                eta_fail=eta_fail,
                deadlock_signal=deadlock_signal,
            )
            active_remedies = list(remedies)
            if tmp_remedy is not None:
                active_remedies.append(tmp_remedy)

            repair_output = repair_plan(
                task_goal=self._task_goal,
                state=state,
                plan=[s.to_dict() for s in self._current_plan.subgoals],
                next_index=self._next_index,
                recent_feedback=recent,
                retrieved_skills=skills,
                active_remedies=active_remedies,
                llm_chat=self.llm_chat,
            )
            if repair_output is None:
                return {"repaired": False, "reason": "llm_failed"}

            new_plan_dicts = assemble_repaired_plan(
                plan=[s.to_dict() for s in self._current_plan.subgoals],
                next_index=self._next_index,
                repair_output=repair_output,
            )
            self._current_plan.subgoals = [
                Subgoal.from_dict(d) for d in new_plan_dicts
            ]
            return {
                "repaired": True,
                "plan": self._current_plan.to_dict(),
                "active_remedies_used": list(repair_output.get("active_remedies_used", [])),
                "deadlock_signal": deadlock_signal,
            }

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def advance_subgoal(self, success: bool) -> Dict[str, Any]:
        with self._lock:
            if self._current_plan is None:
                return {"done": True, "reason": "no_plan"}
            if success:
                self._next_index += 1
            done = self._next_index >= len(self._current_plan.subgoals)
            return {
                "done": done,
                "next_index": self._next_index,
                "plan_length": len(self._current_plan.subgoals),
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "task_goal": self._task_goal,
                "buffer_size": len(self.buffer),
                "store_stats": self.store.stats(),
                "next_index": self._next_index,
                "plan_length": len(self._current_plan.subgoals) if self._current_plan else 0,
            }
