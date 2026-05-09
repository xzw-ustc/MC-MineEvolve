"""Thin adapter to load the STEVE-1 low-level policy.

We do NOT vendor any STEVE-1 source code. Instead the loader tries two
public sources in order:

    1) MineStudio (`pip install MineStudio`):
       ``minestudio.models.load_steve_one_policy(...)`` returns a ready
       ``SteveOnePolicy`` whose ``forward`` produces buttons + camera and
       whose ``prepare_condition`` accepts a {'cond_scale', 'text'} dict.
       Weights are pulled from ``CraftJarvis/MineStudio_STEVE-1.official``
       on HuggingFace at first call.

    2) The original STEVE-1 package (`pip install steve1`):
       loaded via ``steve1.MineRLConditionalAgent`` if MineStudio is not
       installed. The user is expected to have placed weights under
       ``checkpoints/`` per the README.

Both code paths return an object exposing a ``run_step(condition, obs,
state)`` callable used by ``steve_runner.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger("mineevolve.executor.steve_loader")


@dataclass
class SteveCheckpoint:
    in_model: str = "checkpoints/vpt/2x.model"
    in_weights: str = "checkpoints/steve1/steve1.weights"
    prior_weights: str = "checkpoints/steve1/steve1_prior.pt"
    mineclip_weights: str = "checkpoints/mineclip/attn.pth"


def _try_minestudio(ckpt: SteveCheckpoint) -> Optional[Any]:
    try:
        from minestudio.models import load_steve_one_policy  # type: ignore
    except Exception as exc:
        logger.info("MineStudio not available (%s); falling back.", exc)
        return None
    try:
        policy = load_steve_one_policy(
            "CraftJarvis/MineStudio_STEVE-1.official"
        )
    except Exception as exc:
        logger.warning("MineStudio load_steve_one_policy failed: %s", exc)
        return None
    logger.info("Loaded STEVE-1 via MineStudio.")
    return policy


def _try_steve1_pkg(ckpt: SteveCheckpoint) -> Optional[Any]:
    try:
        from steve1.utils.embed_utils import get_prior_embed  # type: ignore  # noqa: F401
        from steve1.config import MINECLIP_CONFIG, PRIOR_INFO  # type: ignore  # noqa: F401
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent  # type: ignore
        from steve1.utils.mineclip_agent_env_utils import (  # type: ignore
            load_model_parameters,
        )
    except Exception as exc:
        logger.info("Original STEVE-1 package not available (%s).", exc)
        return None

    for path in (ckpt.in_model, ckpt.in_weights, ckpt.prior_weights):
        if not os.path.exists(path):
            logger.warning(
                "STEVE-1 weight file missing: %s (set checkpoints/ per README).",
                path,
            )
            return None

    try:
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(ckpt.in_model)
        agent = MineRLConditionalAgent(
            env=None,
            device="cuda",
            policy_kwargs=agent_policy_kwargs,
            pi_head_kwargs=agent_pi_head_kwargs,
        )
        agent.load_weights(ckpt.in_weights)
    except Exception as exc:
        logger.error("STEVE-1 native loader failed: %s", exc)
        return None
    logger.info("Loaded STEVE-1 via the original `steve1` package.")
    return agent


def load_steve_policy(
    in_model: str = "checkpoints/vpt/2x.model",
    in_weights: str = "checkpoints/steve1/steve1.weights",
    prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    mineclip_weights: str = "checkpoints/mineclip/attn.pth",
) -> Any:
    """Load the STEVE-1 policy via the first available backend."""

    ckpt = SteveCheckpoint(
        in_model=in_model,
        in_weights=in_weights,
        prior_weights=prior_weights,
        mineclip_weights=mineclip_weights,
    )
    policy = _try_minestudio(ckpt) or _try_steve1_pkg(ckpt)
    if policy is None:
        raise RuntimeError(
            "Failed to load STEVE-1. Install one of:\n"
            "  - pip install MineStudio   (recommended; auto-downloads weights)\n"
            "  - pip install steve1       (and place weights under checkpoints/)\n"
            "See checkpoints/README.md for details."
        )
    return policy
