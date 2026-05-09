"""GLM-4 backend via Zhipu BigModel's OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Optional

from .openai_compat import OpenAICompatibleBackend


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class GLMBackend(OpenAICompatibleBackend):
    name = "glm"

    def __init__(
        self,
        model: str = "glm-4.7",
        api_key: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_BASE_URL,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="ZHIPUAI_API_KEY",
            **kwargs,
        )
