"""Provider-neutral LLM clients for module-local evidence extraction."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system: str, prompt: str) -> str:
        """Return a provider's text completion."""


class OpenAICompatibleProvider:
    sampling = "temperature=0.0"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("E2_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible extraction")

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


class GeminiProvider:
    sampling = "temperature=0.0"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model = model or os.getenv("E2_LLM_MODEL") or "gemini-3.8-flash"
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini extraction")

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code not in {500, 502, 503, 504} or attempt == 2:
                    raise
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = max(0.0, float(retry_after)) if retry_after else 2**attempt
                except ValueError:
                    delay = 2**attempt
                time.sleep(min(delay, 8.0))
        return result["choices"][0]["message"]["content"]


def provider_from_environment() -> LLMProvider:
    provider = os.getenv("E2_LLM_PROVIDER", "").lower()
    if provider == "openai":
        return OpenAICompatibleProvider()
    if provider == "gemini":
        return GeminiProvider()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAICompatibleProvider()
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()
    raise ValueError(
        "Set OPENAI_API_KEY or GEMINI_API_KEY/GOOGLE_API_KEY before live E2 extraction"
    )
