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
from typing import Any, Mapping, Optional

import numpy as np


logger = logging.getLogger("mineevolve.executor.steve_loader")


@dataclass
class SteveCheckpoint:
    in_model: str = "checkpoints/vpt/2x.model"
    in_weights: str = "checkpoints/steve1/steve1.weights"
    prior_weights: str = "checkpoints/steve1/steve1_prior.pt"
    mineclip_weights: str = "checkpoints/mineclip/attn.pth"


class NativeSteve1Policy:
    """Text-conditioned wrapper for the original STEVE-1 package."""

    def __init__(
        self,
        agent: Any,
        mineclip: Any,
        prior: Any,
        device: str,
        cond_scale: float = 4.0,
    ) -> None:
        self.agent = agent
        self.mineclip = mineclip
        self.prior = prior
        self.device = device
        self.cond_scale = float(cond_scale)
        self._prompt_embeds: dict[str, Any] = {}
        self._condition_text = ""

    def set_text_condition(self, text: str, scale: float | None = None) -> None:
        self._condition_text = str(text)
        if scale is not None:
            self.cond_scale = float(scale)

    def reset_episode(self) -> None:
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset(cond_scale=self.cond_scale)

    def get_action(self, obs: Mapping[str, Any]) -> dict:
        if "pov" not in obs:
            raise RuntimeError("STEVE-1 requires raw MineRL observations with a 'pov' frame.")
        obs = dict(obs)
        if not isinstance(obs["pov"], np.ndarray):
            obs["pov"] = np.asarray(obs["pov"], dtype=np.uint8)
        prompt_embed = self._prompt_embeds.get(self._condition_text)
        if prompt_embed is None:
            prompt_embed = self._embed_text(self._condition_text)
            self._prompt_embeds[self._condition_text] = prompt_embed
        return dict(self.agent.get_action(obs, prompt_embed))

    def _embed_text(self, text: str) -> Any:
        import torch  # type: ignore

        with torch.no_grad():
            text_embed = self.mineclip.encode_text(text).float().detach().cpu()
            prior_input = text_embed.to(device=self.device, dtype=torch.float32)
            return self.prior(prior_input).float().cpu().detach().numpy()


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


def _try_steve1_pkg(
    ckpt: SteveCheckpoint,
    device: str = "cuda",
    cond_scale: float = 4.0,
) -> Optional[Any]:
    try:
        import gym  # type: ignore
        import torch  # type: ignore
        from steve1.config import MINECLIP_CONFIG, PRIOR_INFO  # type: ignore
        from steve1.data.text_alignment.vae import load_vae_model  # type: ignore
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent  # type: ignore
        from steve1.mineclip_code.load_mineclip import load as load_mineclip  # type: ignore
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
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(ckpt.in_model)
        agent_pi_head_kwargs["temperature"] = float(agent_pi_head_kwargs["temperature"])

        bootstrap_env = gym.make("MineRLBasaltFindCave-v0")
        try:
            agent = MineRLConditionalAgent(
                bootstrap_env,
                device=device,
                policy_kwargs=agent_policy_kwargs,
                pi_head_kwargs=agent_pi_head_kwargs,
            )
            agent.load_weights(ckpt.in_weights)
        finally:
            bootstrap_env.close()

        mineclip_config = dict(MINECLIP_CONFIG)
        if os.path.exists(ckpt.mineclip_weights):
            mineclip_config["ckpt"] = dict(mineclip_config.get("ckpt", {}))
            mineclip_config["ckpt"]["path"] = ckpt.mineclip_weights
        mineclip = load_mineclip(mineclip_config, device=device)
        prior_info = dict(PRIOR_INFO)
        prior_info["model_path"] = ckpt.prior_weights
        prior = load_vae_model(prior_info)
    except Exception as exc:
        logger.error("STEVE-1 native loader failed: %s", exc)
        return None
    logger.info("Loaded STEVE-1 via the original `steve1` package.")
    policy = NativeSteve1Policy(
        agent=agent,
        mineclip=mineclip,
        prior=prior,
        device=device,
        cond_scale=cond_scale,
    )
    policy.reset_episode()
    return policy


def load_steve_policy(
    in_model: str = "checkpoints/vpt/2x.model",
    in_weights: str = "checkpoints/steve1/steve1.weights",
    prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    mineclip_weights: str = "checkpoints/mineclip/attn.pth",
    device: str = "cuda",
    cond_scale: float = 4.0,
) -> Any:
    """Load the STEVE-1 policy via the first available backend."""

    ckpt = SteveCheckpoint(
        in_model=in_model,
        in_weights=in_weights,
        prior_weights=prior_weights,
        mineclip_weights=mineclip_weights,
    )
    policy = _try_minestudio(ckpt) or _try_steve1_pkg(
        ckpt,
        device=device,
        cond_scale=cond_scale,
    )
    if policy is None:
        raise RuntimeError(
            "Failed to load STEVE-1. Install one of:\n"
            "  - pip install MineStudio   (recommended; auto-downloads weights)\n"
            "  - pip install steve1       (and place weights under checkpoints/)\n"
            "See checkpoints/README.md for details."
        )
    return policy
