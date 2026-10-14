"""Provider-agnostic LLM client.

Supported providers:
* ``gemini``        - Google Gemini (GEMINI_API_KEY, GEMINI_MODEL)
* ``azure_openai``  - Azure OpenAI (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT)
* ``openai``        - OpenAI-compatible endpoint (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL)
* ``mock``          - deterministic offline writer, used for tests, CI and demos without keys

Calls go over plain HTTPS with ``httpx`` so the project does not depend on vendor SDKs.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

import httpx

from . import config


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, prompt: str) -> str: ...

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        fenced = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.DOTALL)
        return fenced.group(1).strip() if fenced else text


class MockLLM(LLMClient):
    """Renders the memo template from the facts block - no network, fully deterministic."""

    name = "mock"

    def generate(self, system: str, prompt: str) -> str:
        from .memo import render_template_memo

        match = re.search(r"<facts>\s*(\{.*?\})\s*</facts>", prompt, re.DOTALL)
        if not match:
            raise LLMError("Mock LLM expects a <facts>{...}</facts> block in the prompt")
        return render_template_memo(json.loads(match.group(1)))


class GeminiLLM(LLMClient):
    name = "gemini"

    def __init__(self):
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, system: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": config.LLM_TEMPERATURE},
        }
        resp = httpx.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=config.LLM_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
        parts = resp.json()["candidates"][0]["content"]["parts"]
        return self._strip_fences("".join(p.get("text", "") for p in parts))


class _ChatCompletionsLLM(LLMClient):
    url: str
    headers: dict
    model: str | None = None

    def generate(self, system: str, prompt: str) -> str:
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": config.LLM_TEMPERATURE,
        }
        if self.model:
            body["model"] = self.model
        resp = httpx.post(self.url, json=body, headers=self.headers, timeout=config.LLM_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            raise LLMError(f"{self.name} error {resp.status_code}: {resp.text[:300]}")
        return self._strip_fences(resp.json()["choices"][0]["message"]["content"] or "")


class AzureOpenAILLM(_ChatCompletionsLLM):
    name = "azure_openai"

    def __init__(self):
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
        self.headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}


class OpenAILLM(_ChatCompletionsLLM):
    name = "openai"

    def __init__(self):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


PROVIDERS = {"mock": MockLLM, "gemini": GeminiLLM, "azure_openai": AzureOpenAILLM, "openai": OpenAILLM}


def get_llm(provider: str | None = None) -> LLMClient:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "auto":
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            provider = "azure_openai"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from {sorted(PROVIDERS)}.")
    try:
        return PROVIDERS[provider]()
    except KeyError as missing:
        raise LLMError(f"Provider '{provider}' needs environment variable {missing}") from None
