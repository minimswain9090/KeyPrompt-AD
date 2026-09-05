"""Configuration objects for the KeyPrompt-AD pipeline.

Everything the pipeline needs is expressed as a dataclass so that a run can be
serialised alongside its results. Reproducibility of a benchmark run means
being able to point at one YAML file and one commit hash.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProviderConfig:
    """Which vision-language model to call, and how."""

    name: str = "gemini"  # gemini | openrouter | groq | echo
    model: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_retries: int = 4
    retry_base_delay: float = 4.0
    # Free tiers are aggressively rate limited. Pace requests rather than
    # burning the quota on 429s.
    min_seconds_between_calls: float = 4.0
    timeout: float = 120.0


@dataclass
class ImageConfig:
    """How reference and query images are prepared before they are sent."""

    max_side: int = 512
    keep_aspect: bool = True
    jpeg_quality: int = 90


@dataclass
class PriorConfig:
    """Parameters for building the normality graph from the reference shots."""

    n_shots: int = 4
    # Reference annotations are aligned to a canonical frame before averaging.
    align: bool = True
    # Minimum positional sigma (normalised units) so that a class observed in
    # only a handful of shots does not collapse to a zero-variance slot.
    min_sigma: float = 0.012


@dataclass
class ScoringConfig:
    """Fusion weights for the continuous anomaly score.

    The VLM emits a discrete verdict, which alone cannot produce an ROC curve.
    The geometric layer converts the returned keypoints into a continuous
    deviation score, so classification threshold and detection quality can be
    separated cleanly.
    """

    # Gating radius (normalised units) for assigning a detection to a prior slot.
    match_radius: float = 0.10
    w_missing: float = 1.0     # prior slots with no detection nearby
    w_extra: float = 0.5       # detections with no prior slot nearby
    w_displacement: float = 1.0  # normalised offset of matched slots
    w_edge: float = 0.8        # violation of learned pairwise distances
    w_vlm: float = 0.6         # the model's own verdict/confidence
    # Displacement and edge terms are expressed in units of the learned sigma;
    # this caps the influence of one wild outlier.
    sigma_clip: float = 6.0


@dataclass
class CategoryConfig:
    """Per-category description used to build the context prompt."""

    name: str = "pushpins"
    # Human-written statement of what a normal part looks like. This is the
    # only piece of domain knowledge the pipeline needs.
    normality_statement: str = ""
    component_classes: List[str] = field(default_factory=list)
    # Defect modes that must be ignored (surface-level variation, etc).
    ignored_variation: List[str] = field(default_factory=list)
    grouping_description: str = ""


@dataclass
class RunConfig:
    dataset_root: Path = Path("data/mvtec_loco_anomaly_detection")
    output_root: Path = Path("runs")
    # Annotations are input data, not run output. They live outside output_root
    # so that an ablation sweep, which varies output_root per variant, reuses
    # one annotation set instead of demanding a fresh one for every run.
    annotations_root: Path = Path("annotations")
    cache_root: Path = Path(".cache/vlm")
    categories: List[str] = field(default_factory=lambda: ["pushpins"])
    seed: int = 0
    limit_per_split: Optional[int] = None  # useful for smoke tests
    use_cache: bool = True

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "RunConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return RunConfig(
            dataset_root=Path(raw.get("dataset_root", "data/mvtec_loco_anomaly_detection")),
            output_root=Path(raw.get("output_root", "runs")),
            annotations_root=Path(raw.get("annotations_root", "annotations")),
            cache_root=Path(raw.get("cache_root", ".cache/vlm")),
            categories=raw.get("categories", ["pushpins"]),
            seed=int(raw.get("seed", 0)),
            limit_per_split=raw.get("limit_per_split"),
            use_cache=bool(raw.get("use_cache", True)),
            provider=ProviderConfig(**raw.get("provider", {})),
            image=ImageConfig(**raw.get("image", {})),
            prior=PriorConfig(**raw.get("prior", {})),
            scoring=ScoringConfig(**raw.get("scoring", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("dataset_root", "output_root", "cache_root", "annotations_root"):
            d[k] = str(d[k])
        return d


def load_category(path: str | Path) -> CategoryConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return CategoryConfig(**raw)
