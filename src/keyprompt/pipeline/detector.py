"""End-to-end inspection of a single image.

The flow per query image is: assemble prompt -> one VLM call -> parse JSON ->
audit the returned points against the normality graph -> emit a continuous
score and a set of defect coordinates.

There is no training step anywhere in this file, which is the point. The only
per-category cost is annotating K reference images once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..annotate.schema import ImageAnnotation
from ..config import CategoryConfig, RunConfig
from ..data.loco import Sample
from ..prior.graph import NormalityGraph
from ..prompting.builder import build_context_prompt, build_query_prompt, build_shot_block
from ..prompting.schema import InspectionResult, parse_response
from ..providers.base import Content, VLMProvider, prepare_image
from .scoring import ScoreBreakdown, score_detection


@dataclass
class Prediction:
    uid: str
    category: str
    subset: str
    label: int
    score: float
    verdict: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    parse_ok: bool = True
    error: Optional[str] = None
    breakdown: Optional[Dict] = None
    defect_points: List[Tuple[float, float]] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "uid": self.uid,
            "category": self.category,
            "subset": self.subset,
            "label": self.label,
            "score": round(self.score, 5),
            "verdict": self.verdict,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cached": self.cached,
            "parse_ok": self.parse_ok,
            "error": self.error,
            "breakdown": self.breakdown,
            "defect_points": [[round(x, 4), round(y, 4)] for x, y in self.defect_points],
            "reasoning": self.reasoning,
        }


class KeyPromptDetector:
    def __init__(
        self,
        cfg: RunConfig,
        category: CategoryConfig,
        graph: NormalityGraph,
        reference_images: List[Image.Image],
        reference_annotations: List[ImageAnnotation],
        provider: VLMProvider,
    ):
        if len(reference_images) != len(reference_annotations):
            raise ValueError("reference images and annotations must align one-to-one")
        self.cfg = cfg
        self.category = category
        self.graph = graph
        self.provider = provider
        self.ref_images = [prepare_image(im, cfg.image) for im in reference_images]
        self.ref_annotations = reference_annotations
        self._context = build_context_prompt(category, graph)
        self._query = build_query_prompt()

    # -- prompt ---------------------------------------------------------

    def _contents(self, query_image: Image.Image) -> List[Content]:
        contents: List[Content] = [self._context, ""]
        for i, (img, ann) in enumerate(zip(self.ref_images, self.ref_annotations), start=1):
            contents.append(build_shot_block(i, ann))
            contents.append(img)
        contents.append(self._query)
        contents.append(prepare_image(query_image, self.cfg.image))
        return contents

    # -- inference ------------------------------------------------------

    def predict(self, sample: Sample) -> Prediction:
        t0 = time.time()
        img = sample.load_image()
        resp = self.provider.generate(self._contents(img))

        if resp.error:
            return Prediction(
                uid=sample.uid,
                category=sample.category,
                subset=sample.subset,
                label=sample.label,
                score=0.5,  # neutral: an API failure is not evidence either way
                verdict="ERROR",
                latency_s=time.time() - t0,
                cached=resp.cached,
                parse_ok=False,
                error=resp.error,
            )

        try:
            parsed: InspectionResult = parse_response(resp.text)
            parse_ok = True
            err = None
        except Exception as exc:  # noqa: BLE001
            parsed = InspectionResult(verdict="OK", confidence=0.5)
            parse_ok = False
            err = f"parse failure: {exc}"

        detected: Dict[str, List[Tuple[float, float]]] = {}
        for d in parsed.detected:
            detected.setdefault(d.cls, []).append((d.x, d.y))

        breakdown: ScoreBreakdown = score_detection(
            detected=detected,
            graph=self.graph,
            cfg=self.cfg.scoring,
            vlm_verdict=parsed.verdict,
            vlm_confidence=parsed.confidence,
        )

        # Points the model explicitly reported as missing are localisation
        # evidence in their own right and are merged with the geometric ones.
        pts = list(breakdown.defect_points()) + [(m.x, m.y) for m in parsed.missing]

        return Prediction(
            uid=sample.uid,
            category=sample.category,
            subset=sample.subset,
            label=sample.label,
            score=breakdown.score,
            verdict=parsed.verdict,
            latency_s=resp.latency_s if resp.latency_s else time.time() - t0,
            prompt_tokens=resp.prompt_tokens,
            output_tokens=resp.output_tokens,
            cached=resp.cached,
            parse_ok=parse_ok,
            error=err,
            breakdown=breakdown.to_dict(),
            defect_points=_dedupe(pts),
            reasoning=parsed.reasoning,
        )


def _dedupe(points: List[Tuple[float, float]], tol: float = 0.01) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        if all(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol for q in out):
            out.append(p)
    return out
