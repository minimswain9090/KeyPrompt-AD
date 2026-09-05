"""Provider abstraction over vision-language APIs.

Every backend receives the same interleaved sequence of text and images and
returns raw text. Keeping the interface this thin means swapping Gemini for a
Qwen checkpoint on OpenRouter is a one-line config change, which is what makes
the cross-model ablation in the evaluation cheap to run.

Three concerns live here rather than in the individual backends:

* a content-addressed response cache, so a rerun costs nothing and results stay
  reproducible after the free tier resets;
* retry with exponential backoff, because free tiers return 429 constantly;
* a request pacer, for the same reason.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..config import ImageConfig, ProviderConfig

Content = Union[str, Image.Image]


@dataclass
class VLMResponse:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def prepare_image(img: Image.Image, cfg: ImageConfig) -> Image.Image:
    """Downscale before upload.

    Resolution is the dominant cost and latency driver. It is also a genuine
    experimental variable: how small can the input get before the localisation
    degrades? That ablation is reported in the evaluation.
    """
    out = img.convert("RGB")
    if cfg.keep_aspect:
        out.thumbnail((cfg.max_side, cfg.max_side), Image.Resampling.LANCZOS)
    else:
        out = out.resize((cfg.max_side, cfg.max_side), Image.Resampling.LANCZOS)
    return out


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLMProvider(ABC):
    """Base class handling caching, pacing and retries."""

    def __init__(self, cfg: ProviderConfig, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_at = 0.0

    # -- subclass hook --------------------------------------------------

    @abstractmethod
    def _call(self, contents: List[Content]) -> VLMResponse:
        """Issue one request. Raise on failure; retries are handled upstream."""

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"environment variable {self.cfg.api_key_env} is not set.\n"
                f"Fix: copy .env.example to .env in the repository root and add "
                f"a line reading  {self.cfg.api_key_env}=<your key>\n"
                f"The .env file is loaded automatically and is gitignored."
            )
        return key

    # -- public API -----------------------------------------------------

    def generate(self, contents: List[Content]) -> VLMResponse:
        key = self._cache_key(contents)
        cached = self._cache_read(key)
        if cached is not None:
            return cached

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            self._pace()
            try:
                resp = self._call(contents)
                self._cache_write(key, resp)
                return resp
            except Exception as exc:  # noqa: BLE001 - surfaced in the response
                last_err = exc
                delay = self.cfg.retry_base_delay * (2 ** attempt)
                if attempt < self.cfg.max_retries - 1:
                    print(f"  [{self.cfg.name}] attempt {attempt + 1} failed ({exc}); "
                          f"retrying in {delay:.0f}s")
                    time.sleep(delay)

        return VLMResponse(text="", latency_s=0.0, error=str(last_err))

    # -- internals ------------------------------------------------------

    def _pace(self) -> None:
        gap = time.time() - self._last_call_at
        wait = self.cfg.min_seconds_between_calls - gap
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.time()

    def _cache_key(self, contents: List[Content]) -> str:
        h = hashlib.sha256()
        h.update(f"{self.cfg.name}|{self.cfg.model}|{self.cfg.temperature}|{self.cfg.max_output_tokens}".encode())
        for c in contents:
            if isinstance(c, str):
                h.update(b"T")
                h.update(c.encode())
            else:
                h.update(b"I")
                buf = io.BytesIO()
                c.save(buf, format="PNG")
                h.update(hashlib.sha256(buf.getvalue()).digest())
        return h.hexdigest()[:32]

    def _cache_read(self, key: str) -> Optional[VLMResponse]:
        if not (self.use_cache and self.cache_dir):
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return VLMResponse(
            text=d["text"],
            latency_s=d.get("latency_s", 0.0),
            prompt_tokens=d.get("prompt_tokens"),
            output_tokens=d.get("output_tokens"),
            cached=True,
            error=d.get("error"),
        )

    def _cache_write(self, key: str, resp: VLMResponse) -> None:
        if not (self.use_cache and self.cache_dir) or resp.error or not resp.text.strip():
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(
                {
                    "text": resp.text,
                    "latency_s": resp.latency_s,
                    "prompt_tokens": resp.prompt_tokens,
                    "output_tokens": resp.output_tokens,
                },
                indent=2,
            )
        )
