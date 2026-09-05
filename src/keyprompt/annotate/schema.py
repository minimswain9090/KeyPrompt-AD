"""Annotation format for reference (OK) images.

A reference annotation is deliberately sparse: a handful of labelled points and
an optional group id per point. This is what a line engineer can produce in a
couple of minutes per part, and it is the entire supervision signal the method
receives.

Coordinates are normalised to [0, 1] with the origin at the top-left corner, so
annotations survive any resizing applied before inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Keypoint:
    cls: str                      # component class, e.g. "pushpin"
    x: float                      # normalised, [0, 1]
    y: float                      # normalised, [0, 1]
    group: Optional[str] = None   # e.g. "compartment_A"; enables shape checks
    note: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ImageAnnotation:
    image_uid: str
    category: str
    width: int
    height: int
    keypoints: List[Keypoint] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k in self.keypoints:
            out[k.cls] = out.get(k.cls, 0) + 1
        return out

    def by_class(self, cls: str) -> List[Keypoint]:
        return [k for k in self.keypoints if k.cls == cls]

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "ImageAnnotation":
        raw = json.loads(Path(path).read_text())
        kps = [Keypoint(**k) for k in raw.pop("keypoints", [])]
        return ImageAnnotation(keypoints=kps, **raw)


def load_annotation_set(directory: str | Path) -> List[ImageAnnotation]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return [ImageAnnotation.from_json(p) for p in sorted(d.glob("*.json"))]
