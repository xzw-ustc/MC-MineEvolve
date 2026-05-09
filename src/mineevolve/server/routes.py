"""HTTP request models for the FastAPI app."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    task_goal: str


class ChatRequest(BaseModel):
    type: str = Field(..., description="One of plan|action|monitor|induce|repair|advance|status")
    payload: Dict[str, Any] = Field(default_factory=dict)


class PlanPayload(BaseModel):
    state: Dict[str, Any] = Field(default_factory=dict)
    budget_tokens: int = 512
    top_k: int = 16


class ActionPayload(BaseModel):
    condition: str
    obs: Dict[str, Any] = Field(default_factory=dict)


class MonitorPayload(BaseModel):
    subgoal: List[Any]
    target_item: Optional[str] = None
    coords_history: Sequence[Sequence[float]] = Field(default_factory=list)
    delta_v: Dict[str, int] = Field(default_factory=dict)
    delta_s: Dict[str, Any] = Field(default_factory=dict)
    gui_open: bool = False
    success: int = 0
    timed_out: bool = False
    died: bool = False
    p_goal: float = 0.0
    subgoal_id: Optional[str] = None
    coords_now: Optional[Sequence[float]] = None


class InducePayload(BaseModel):
    eta_fail: float = 0.5
    recent_window: int = 4
    state: Dict[str, Any] = Field(default_factory=dict)
    deadlock_window: int = 8
    deadlock_min_distinct_subgoals: int = 2
    deadlock_min_repeats: int = 3


class RepairPayload(BaseModel):
    state: Dict[str, Any] = Field(default_factory=dict)
    recent_window: int = 4
    eta_fail: float = 0.5
    budget_tokens: int = 512
    top_k: int = 16
    deadlock_window: int = 8
    deadlock_min_distinct_subgoals: int = 2
    deadlock_min_repeats: int = 3


class AdvancePayload(BaseModel):
    success: bool
