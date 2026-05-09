"""Generic OpenAI-compatible chat backend.

Used directly for the OpenAI GPT family and as the base class for Qwen
(via DashScope's OpenAI-compatible endpoint), GLM (via Zhipu BigModel's
OpenAI-compatible endpoint), and Gemini (via Google AI Studio's
OpenAI-compatible endpoint).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..base import PlannerBackend


logger = logging.getLogger("mineevolve.planner.openai_compat")


class OpenAICompatibleBackend(PlannerBackend):
    """Thin wrapper around the official ``openai`` python SDK.

    The vendor is selected solely by ``base_url`` and ``api_key``.
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `openai` package is required. Install it with `pip install openai`."
            ) from exc

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            logger.warning(
                "API key not set (env=%s). Backend %s will fail at call time.",
                api_key_env,
                self.name,
            )
        self._client = OpenAI(
            api_key=resolved_key or "missing",
            base_url=base_url,
            timeout=timeout,
        )
        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type(Exception),
    )
    def _do_chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content if choice and choice.message else ""
        return content or ""

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 0,
        temperature: float = -1.0,
    ) -> str:
        if max_tokens <= 0:
            max_tokens = self._default_max_tokens
        if temperature < 0:
            temperature = self._default_temperature
        try:
            return self._do_chat(system, user, max_tokens, temperature)
        except Exception as exc:
            logger.error("LLM call failed (%s/%s): %s", self.name, self._model, exc)
            return ""
