"""Gemini backend via Google AI Studio's OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Optional

from .openai_compat import OpenAICompatibleBackend


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiBackend(OpenAICompatibleBackend):
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_BASE_URL,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="GOOGLE_API_KEY",
            **kwargs,
        )
