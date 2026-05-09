"""Pluggable LLM backends for the MineEvolve planner.

Resolution helper ``get_backend(cfg)`` reads a Hydra ``llm`` config and
returns a ready-to-call ``PlannerBackend`` instance.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..base import PlannerBackend
from .gemini import GeminiBackend
from .glm import GLMBackend
from .gpt import OpenAIBackend
from .openai_compat import OpenAICompatibleBackend
from .qwen import QwenBackend


_REGISTRY: dict[str, type[PlannerBackend]] = {
    "openai": OpenAIBackend,
    "gpt": OpenAIBackend,
    "qwen": QwenBackend,
    "glm": GLMBackend,
    "gemini": GeminiBackend,
    "openai_compat": OpenAICompatibleBackend,
}


def get_backend(cfg: Mapping[str, Any]) -> PlannerBackend:
    """Build a planner backend from a Hydra ``llm`` config.

    Expected keys in ``cfg``:
        provider:    one of {openai, qwen, glm, gemini, openai_compat}
        model:       model name string
        api_key:     optional explicit key (else read from env)
        base_url:    optional override
        timeout:     seconds
        temperature: float
        max_tokens:  int
    """

    provider = str(cfg.get("provider") or "openai").lower()
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Available: {sorted(_REGISTRY)}"
        )

    cls = _REGISTRY[provider]
    return cls(
        model=str(cfg.get("model") or ""),
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        timeout=float(cfg.get("timeout", 120.0)),
        default_temperature=float(cfg.get("temperature", 0.2)),
        default_max_tokens=int(cfg.get("max_tokens", 1024)),
    )


__all__ = [
    "GeminiBackend",
    "GLMBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "QwenBackend",
    "PlannerBackend",
    "get_backend",
]
