"""FastAPI app exposing /reset and /chat for the MineEvolve agent.

Run with `uvicorn mineevolve.server.api:create_app --factory --port 9000`
or via the helper script in scripts/server.sh.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException

from .agent import MineEvolveAgent
from .routes import (
    ActionPayload,
    AdvancePayload,
    ChatRequest,
    InducePayload,
    MonitorPayload,
    PlanPayload,
    RepairPayload,
    ResetRequest,
)


logger = logging.getLogger("mineevolve.server.api")


def create_app(cfg: Mapping[str, Any] | None = None) -> FastAPI:
    """Build a FastAPI app with one global ``MineEvolveAgent`` instance."""

    app = FastAPI(title="MineEvolve Server", version="0.1.0")
    app.state.cfg = dict(cfg or _bootstrap_default_cfg())
    app.state.agent = MineEvolveAgent(cfg=app.state.cfg, logger=logger)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/reset")
    def reset(req: ResetRequest) -> dict:
        return app.state.agent.reset(task_goal=req.task_goal)

    @app.get("/status")
    def status() -> dict:
        return app.state.agent.status()

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        agent: MineEvolveAgent = app.state.agent
        t = (req.type or "").lower()
        try:
            if t == "plan":
                p = PlanPayload(**req.payload)
                return agent.plan(state=p.state, budget_tokens=p.budget_tokens, top_k=p.top_k)
            if t == "action":
                p = ActionPayload(**req.payload)
                return agent.action(condition=p.condition, obs=p.obs)
            if t == "monitor":
                p = MonitorPayload(**req.payload)
                return agent.monitor_subgoal(
                    subgoal=p.subgoal,
                    target_item=p.target_item,
                    coords_history=p.coords_history,
                    delta_v=p.delta_v,
                    delta_s=p.delta_s,
                    gui_open=p.gui_open,
                    success=p.success,
                    timed_out=p.timed_out,
                    died=p.died,
                    p_goal=p.p_goal,
                    subgoal_id=p.subgoal_id,
                    coords_now=p.coords_now,
                )
            if t == "induce":
                p = InducePayload(**req.payload)
                return agent.induce_and_curate(
                    eta_fail=p.eta_fail,
                    recent_window=p.recent_window,
                    deadlock_window=p.deadlock_window,
                    deadlock_min_distinct_subgoals=p.deadlock_min_distinct_subgoals,
                    deadlock_min_repeats=p.deadlock_min_repeats,
                    state=p.state,
                )
            if t == "repair":
                p = RepairPayload(**req.payload)
                return agent.repair(
                    state=p.state,
                    recent_window=p.recent_window,
                    eta_fail=p.eta_fail,
                    budget_tokens=p.budget_tokens,
                    top_k=p.top_k,
                    deadlock_window=p.deadlock_window,
                    deadlock_min_distinct_subgoals=p.deadlock_min_distinct_subgoals,
                    deadlock_min_repeats=p.deadlock_min_repeats,
                )
            if t == "advance":
                p = AdvancePayload(**req.payload)
                return agent.advance_subgoal(success=p.success)
            if t == "status":
                return agent.status()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("chat error (type=%s)", t)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        raise HTTPException(status_code=400, detail=f"unknown chat type: {req.type!r}")

    return app


def _bootstrap_default_cfg() -> dict:
    """Minimal default config used when the server is started without one.

    Default LLM backend is Qwen Plus via DashScope's OpenAI-compatible
    endpoint. Override by setting MINEEVOLVE_LLM_PROVIDER /
    MINEEVOLVE_LLM_MODEL / MINEEVOLVE_LLM_BASE_URL or by passing your own
    cfg into ``create_app``.
    """

    return {
        "monitor": {},
        "memory": {"path": os.environ.get("MINEEVOLVE_MEMORY_PATH", "memories/run")},
        "llm": {
            "provider": os.environ.get("MINEEVOLVE_LLM_PROVIDER", "qwen"),
            "model": os.environ.get("MINEEVOLVE_LLM_MODEL", "qwen-plus"),
            "base_url": os.environ.get(
                "MINEEVOLVE_LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "api_key": os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "temperature": 0.2,
            "max_tokens": 1536,
        },
        "steve": {
            "in_model": os.environ.get(
                "MINEEVOLVE_VPT_MODEL", "checkpoints/vpt/2x.model"
            ),
            "in_weights": os.environ.get(
                "MINEEVOLVE_STEVE_WEIGHTS", "checkpoints/steve1/steve1.weights"
            ),
            "prior_weights": os.environ.get(
                "MINEEVOLVE_STEVE_PRIOR", "checkpoints/steve1/steve1_prior.pt"
            ),
            "mineclip_weights": os.environ.get(
                "MINEEVOLVE_MINECLIP_WEIGHTS", "checkpoints/mineclip/attn.pth"
            ),
        },
        "load_steve": True,
    }
