"""Concrete backends.

All four are usable at zero cost at the time of writing, though free-tier terms
move around; check current limits before planning a full sweep.

  gemini      Google AI Studio. Native structured output, generous free tier,
              strong at spatial grounding. Default choice.
  openrouter  Aggregator exposing several open-weight VLMs on a free tier,
              e.g. Qwen2.5-VL and Llama vision variants. Useful for showing the
              method is not tied to one proprietary model.
  groq        Fast inference on open-weight vision models; the lowest-latency
              option, which matters for the throughput argument.
  echo        Offline stub. Returns a fixed empty detection so the pipeline,
              scorer and metrics can be exercised without network access.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from PIL import Image

from ..prompting.schema import JSON_SCHEMA_HINT
from .base import Content, VLMProvider, VLMResponse, image_to_b64


class GeminiProvider(VLMProvider):
    def _call(self, contents: List[Content]) -> VLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        t0 = time.time()
        resp = client.models.generate_content(
            model=self.cfg.model,
            contents=list(contents),
            config=types.GenerateContentConfig(
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        dt = time.time() - t0
        usage = getattr(resp, "usage_metadata", None)
        return VLMResponse(
            text=resp.text or "",
            latency_s=dt,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )


class _OpenAICompatProvider(VLMProvider):
    """Shared implementation for OpenAI-style chat endpoints."""

    base_url = ""
    extra_headers: Dict[str, str] = {}

    def _call(self, contents: List[Content]) -> VLMResponse:
        import requests

        parts: List[Dict[str, Any]] = []
        for c in contents:
            if isinstance(c, str):
                parts.append({"type": "text", "text": c})
            elif isinstance(c, Image.Image):
                b64 = image_to_b64(c)
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )

        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        t0 = time.time()
        r = requests.post(
            self.base_url, headers=headers, data=json.dumps(body), timeout=self.cfg.timeout
        )
        dt = time.time() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        payload = r.json()
        usage = payload.get("usage", {})
        return VLMResponse(
            text=payload["choices"][0]["message"]["content"] or "",
            latency_s=dt,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class OpenRouterProvider(_OpenAICompatProvider):
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    extra_headers = {"X-Title": "KeyPrompt-AD"}


class GroqProvider(_OpenAICompatProvider):
    base_url = "https://api.groq.com/openai/v1/chat/completions"


class EchoProvider(VLMProvider):
    """Offline stub for smoke tests and CI."""

    @property
    def api_key(self) -> str:  # no key needed
        return "offline"

    def _call(self, contents: List[Content]) -> VLMResponse:
        time.sleep(0.01)
        return VLMResponse(
            text=json.dumps(
                {"verdict": "OK", "confidence": 0.0, "detected": [], "missing": [],
                 "reasoning": "offline stub response"}
            ),
            latency_s=0.01,
            prompt_tokens=0,
            output_tokens=0,
        )


_REGISTRY = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "echo": EchoProvider,
}


def build_provider(cfg, cache_dir=None, use_cache: bool = True) -> VLMProvider:
    name = cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown provider '{cfg.name}'. Options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, cache_dir=cache_dir, use_cache=use_cache)


__all__ = ["build_provider", "JSON_SCHEMA_HINT"]
