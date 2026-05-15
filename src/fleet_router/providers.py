"""
fleet_router/providers.py — Model provider adapters.

Each provider has a unified interface: call(prompt, model, temperature, max_tokens) → response.
The router doesn't know about API differences — just asks for a completion.
"""

from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

import httpx


@dataclass
class CompletionResult:
    """Unified response from any model provider."""
    content: str
    reasoning_content: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def text(self) -> str:
        """Get the actual answer text.

        For thinking models (F13), the answer is in reasoning_content.
        For non-thinking models, it's in content.
        """
        return self.content if self.content.strip() else self.reasoning_content


class Provider(ABC):
    """Base class for model providers."""
    name: str = ""

    @abstractmethod
    async def complete(self, prompt: str, model_id: str,
                       temperature: float = 0.0,
                       max_tokens: int = 1024,
                       system: str = "") -> CompletionResult:
        ...

    def _cost(self, tokens_in: int, tokens_out: int, cost_per_1k: float) -> float:
        return (tokens_in + tokens_out) / 1000 * cost_per_1k


class DeepInfraProvider(Provider):
    """DeepInfra API — seed-mini, gemini-lite, hermes, qwen, mimo, step."""
    name = "deepinfra"

    def __init__(self):
        key_path = os.path.expanduser(
            "~/.openclaw/workspace/.credentials/deepinfra-api-key.txt"
        )
        with open(key_path) as f:
            self.api_key = f.read().strip()
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"

    async def complete(self, prompt: str, model_id: str,
                       temperature: float = 0.0,
                       max_tokens: int = 1024,
                       system: str = "") -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
            latency_ms = (time.time() - t0) * 1000
            data = r.json()

            if r.status_code != 200:
                return CompletionResult(
                    content="", model=model_id, provider=self.name,
                    latency_ms=latency_ms, success=False,
                    error=data.get("error", {}).get("message", str(data)),
                )

            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})

            return CompletionResult(
                content=choice.get("content", ""),
                reasoning_content=choice.get("reasoning_content", ""),
                model=model_id,
                provider=self.name,
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                cost=self._cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    0.05,  # average DeepInfra cost
                ),
                success=True,
            )
        except Exception as e:
            return CompletionResult(
                content="", model=model_id, provider=self.name,
                latency_ms=(time.time() - t0) * 1000,
                success=False, error=str(e),
            )


class ZaiProvider(Provider):
    """z.ai API — glm-5-turbo, glm-5.1."""
    name = "zai"

    def __init__(self):
        # z.ai key from OpenClaw config
        self.api_key = "703f56774c324a76b8a283ce50b15744.tLKi6d9yeYza5Spg"
        self.base_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"

    async def complete(self, prompt: str, model_id: str,
                       temperature: float = 0.0,
                       max_tokens: int = 1024,
                       system: str = "") -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
            latency_ms = (time.time() - t0) * 1000
            data = r.json()

            if r.status_code != 200:
                return CompletionResult(
                    content="", model=model_id, provider=self.name,
                    latency_ms=latency_ms, success=False,
                    error=str(data),
                )

            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})

            return CompletionResult(
                content=choice.get("content", ""),
                reasoning_content=choice.get("reasoning_content", ""),
                model=model_id,
                provider=self.name,
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                cost=self._cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    0.08,
                ),
                success=True,
            )
        except Exception as e:
            return CompletionResult(
                content="", model=model_id, provider=self.name,
                latency_ms=(time.time() - t0) * 1000,
                success=False, error=str(e),
            )


class GroqProvider(Provider):
    """Groq API — llama-8b, llama-70b, llama-4-scout, qwen3-32b."""
    name = "groq"

    def __init__(self):
        key_path = os.path.expanduser(
            "~/.openclaw/workspace/.credentials/groq-api-key.txt"
        )
        with open(key_path) as f:
            self.api_key = f.read().strip()
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def complete(self, prompt: str, model_id: str,
                       temperature: float = 0.0,
                       max_tokens: int = 1024,
                       system: str = "") -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
            latency_ms = (time.time() - t0) * 1000
            data = r.json()

            if r.status_code != 200:
                return CompletionResult(
                    content="", model=model_id, provider=self.name,
                    latency_ms=latency_ms, success=False,
                    error=str(data),
                )

            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})

            return CompletionResult(
                content=choice.get("content", ""),
                reasoning_content=choice.get("reasoning_content", ""),
                model=model_id,
                provider=self.name,
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                cost=self._cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    0.01,  # Groq is cheap
                ),
                success=True,
            )
        except Exception as e:
            return CompletionResult(
                content="", model=model_id, provider=self.name,
                latency_ms=(time.time() - t0) * 1000,
                success=False, error=str(e),
            )


# ─── Provider Registry ────────────────────────────────────────────────────────

PROVIDERS: dict[str, Provider] = {}


def get_provider(name: str) -> Provider:
    """Get or create a provider by name."""
    if name not in PROVIDERS:
        if name == "deepinfra":
            PROVIDERS[name] = DeepInfraProvider()
        elif name == "zai":
            PROVIDERS[name] = ZaiProvider()
        elif name == "groq":
            PROVIDERS[name] = GroqProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")
    return PROVIDERS[name]
