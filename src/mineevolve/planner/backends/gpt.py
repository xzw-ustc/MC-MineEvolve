"""OpenAI GPT family backend (e.g. gpt-5.5)."""

from __future__ import annotations

from typing import Optional

from .openai_compat import OpenAICompatibleBackend


class OpenAIBackend(OpenAICompatibleBackend):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="OPENAI_API_KEY",
            **kwargs,
        )
