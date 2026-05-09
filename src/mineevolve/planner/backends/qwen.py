"""Qwen backend via DashScope's OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Optional

from .openai_compat import OpenAICompatibleBackend


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenBackend(OpenAICompatibleBackend):
    name = "qwen"

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_BASE_URL,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="DASHSCOPE_API_KEY",
            **kwargs,
        )
