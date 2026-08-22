"""
Groq LLM client.

This module is responsible only for communicating with the Groq API.
It does not contain prompts, Jaanvi knowledge, routing, or response logic.
"""

from groq import Groq

from app.config import get_settings


class LLMClient:
    """Thin wrapper around the Groq API client."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        if not settings.groq_model:
            raise RuntimeError("GROQ_MODEL is not configured.")

        self.client = Groq(
            api_key=settings.groq_api_key,
        )
        self.model = settings.groq_model
        self.max_tokens = settings.groq_max_tokens

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a chat completion request to Groq
        and return the response text.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
        )

        if not response.choices:
            raise RuntimeError("Groq returned no response choices.")

        message = response.choices[0].message
        content = message.content

        if content is None:
            raise RuntimeError(
                "Groq returned no text content. "
                f"Finish reason: {response.choices[0].finish_reason}"
            )

        return content.strip()