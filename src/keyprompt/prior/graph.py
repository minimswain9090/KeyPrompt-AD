"""The normality graph: a compact statistical model of what "OK" looks like.

Built from K annotated reference images (K is typically 4). It stores, per
component class:

* the expected number of instances,
* canonical slot positions in a common frame,
* per-slot positional spread,

plus the pairwise distances between slots, with their spread. The edge
statistics are what let the system flag a part whose components are all present
but spaced wrongly, which is the failure mode that pure presence-checking
misses.

The graph plays two roles. It is injected into the prompt as an explicit layout
specification, and it is used afterwards to convert the model's returned points
into a continuous deviation score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..annotate.schema import ImageAnnotation
from .geometry import (
    align_by_class,
    apply_transform,
    match_points,
    similarity_transform,
)


@dataclass
class ClassPrior:
    cls: str
    expected_count: int
    slots: np.ndarray            # (M, 2) canonical positions, normalised
    slot_sigma: np.ndarray       # (M,) positional spread per slot
    groups: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cls": self.cls,
            "expected_count": self.expected_count,
            "slots": self.slots.round(4).tolist(),
            "slot_sigma": self.slot_sigma.round(4).tolist(),
            "groups": self.groups,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ClassPrior":
        return ClassPrior(
            cls=d["cls"],
            expected_count=int(d["expected_count"]),
            slots=np.array(d["slots"], dtype=float).reshape(-1, 2),
            slot_sigma=np.array(d["slot_sigma"], dtype=float).reshape(-1),
            groups=d.get("groups", []),
        )


@dataclass
class NormalityGraph:
    category: str
    n_shots: int
    classes: Dict[str, ClassPrior] = field(default_factory=dict)
    # Flattened slot table so edges can index across classes.
    slot_index: List[Tuple[str, int]] = field(default_factory=list)
    edge_mean: Optional[np.ndarray] = None   # (S, S)
    edge_sigma: Optional[np.ndarray] = None  # (S, S)
    source_uids: List[str] = field(default_factory=list)

    # -- construction ---------------------------------------------------

    @staticmethod
    def build(
        annotations: List[ImageAnnotation],
        category: str,
        align: bool = True,
        min_sigma: float = 0.012,
    ) -> "NormalityGraph":
        if not annotations:
            raise ValueError("no reference annotations supplied")

        # Anchor on the most complete reference, not simply the first. If the
        # anchor is missing a component, that slot is absent from the canonical
        # frame and no later shot can reintroduce it -- a silent, permanent hole
        # in the prior. This matters most for automatically proposed
        # annotations, where any single frame may have dropped a detection.
        anchor_idx = int(np.argmax([len(a.keypoints) for a in annotations]))
        ref = annotations[anchor_idx]
        ordered = [ref] + [a for i, a in enumerate(annotations) if i != anchor_idx]
        ref_by_cls = _points_by_class(ref)

        aligned: List[Dict[str, np.ndarray]] = [ref_by_cls]
        for ann in ordered[1:]:
            cur = _points_by_class(ann)
            if align:
                fit = align_by_class(cur, ref_by_cls, radius=0.12)
                if fit is not None:
                    R, s, t = fit
                    cur = {c: apply_transform(p, R, s, t) for c, p in cur.items()}
            aligned.append(cur)

        classes: Dict[str, ClassPrior] = {}
        all_names = sorted({c for a in aligned for c in a})
        for cls in all_names:
            anchor = ref_by_cls.get(cls, np.zeros((0, 2)))
            if len(anchor) == 0:
                # Class absent from the anchor shot: fall back to the first
                # reference that contains it.
                for a in aligned:
                    if len(a.get(cls, [])) > 0:
                        anchor = a[cls]
                        break
            m = len(anchor)
            if m == 0:
                continue

            stacks = [[] for _ in range(m)]
            counts: List[int] = []
            for a in aligned:
                pts = a.get(cls, np.zeros((0, 2)))
                counts.append(len(pts))
                if len(pts) == 0:
                    continue
                pairs, _, _ = match_points(pts, anchor, radius=0.15)
                for pi, si in pairs:
                    stacks[si].append(pts[pi])

            slots = np.zeros((m, 2))
            sigma = np.zeros(m)
            for i in range(m):
                if stacks[i]:
                    arr = np.array(stacks[i])
                    slots[i] = arr.mean(0)
                    sigma[i] = max(float(arr.std(0).mean()), min_sigma)
                else:
                    slots[i] = anchor[i]
                    sigma[i] = min_sigma * 2.0

            groups = _groups_for(ref, cls, m)
            classes[cls] = ClassPrior(
                cls=cls,
                expected_count=int(round(float(np.median(counts)))),
                slots=slots,
                slot_sigma=sigma,
                groups=groups,
            )

        g = NormalityGraph(
            category=category,
            n_shots=len(annotations),
            classes=classes,
            source_uids=[a.image_uid for a in ordered],
        )
        g._build_edges(aligned)
        return g

    def _build_edges(self, aligned: List[Dict[str, np.ndarray]]) -> None:
        self.slot_index = [
            (cls, i) for cls in sorted(self.classes) for i in range(len(self.classes[cls].slots))
        ]
        S = len(self.slot_index)
        if S < 2:
            self.edge_mean = np.zeros((0, 0))
            self.edge_sigma = np.zeros((0, 0))
            return

        per_shot: List[np.ndarray] = []
        for a in aligned:
            coords = np.full((S, 2), np.nan)
            for cls in sorted(self.classes):
                pts = a.get(cls, np.zeros((0, 2)))
                if len(pts) == 0:
                    continue
                pairs, _, _ = match_points(pts, self.classes[cls].slots, radius=0.15)
                base = self.slot_index.index((cls, 0))
                for pi, si in pairs:
                    coords[base + si] = pts[pi]
            per_shot.append(coords)

        D = np.full((len(per_shot), S, S), np.nan)
        for k, coords in enumerate(per_shot):
            diff = coords[:, None, :] - coords[None, :, :]
            D[k] = np.linalg.norm(diff, axis=-1)

        with np.errstate(invalid="ignore"):
            self.edge_mean = np.nanmean(D, axis=0)
            self.edge_sigma = np.nanstd(D, axis=0)
        self.edge_mean = np.nan_to_num(self.edge_mean, nan=0.0)
        self.edge_sigma = np.nan_to_num(self.edge_sigma, nan=0.0)
        # Distances measured from few shots are optimistically tight; floor the
        # spread so a single-pixel wobble is not scored as a five-sigma event.
        self.edge_sigma = np.maximum(self.edge_sigma, 0.015)

    # -- accessors ------------------------------------------------------

    def expected_counts(self) -> Dict[str, int]:
        return {c: p.expected_count for c, p in self.classes.items()}

    def slots_by_class(self) -> Dict[str, np.ndarray]:
        return {c: p.slots for c, p in self.classes.items()}

    def total_slots(self) -> int:
        return sum(len(p.slots) for p in self.classes.values())

    def describe(self, max_neighbours: int = 3) -> str:
        """Plain-language layout specification injected into the prompt.

        The full edge set grows quadratically with the number of slots, which
        would dominate the prompt for a category like pushpins. Only the
        ``max_neighbours`` closest spacings per slot are listed: they carry
        almost all of the constraint, since a component displaced far enough to
        break a long-range distance will already have broken a short-range one.
        """
        lines: List[str] = []
        for cls in sorted(self.classes):
            p = self.classes[cls]
            lines.append(f"- {cls}: expected count = {p.expected_count}")
            for i, (xy, sg) in enumerate(zip(p.slots, p.slot_sigma)):
                grp = p.groups[i] if i < len(p.groups) and p.groups[i] else "-"
                lines.append(
                    f"    slot {cls}[{i}] at (x={xy[0]:.3f}, y={xy[1]:.3f}) "
                    f"tolerance ~{sg:.3f}, group {grp}"
                )
        if self.edge_mean is not None and len(self.slot_index) >= 2:
            shown: set[tuple[int, int]] = set()
            for a in range(len(self.slot_index)):
                row = self.edge_mean[a].copy()
                row[a] = np.inf
                row[row <= 0] = np.inf
                order = np.argsort(row)[:max_neighbours]
                for b in order:
                    if not np.isfinite(row[b]):
                        continue
                    shown.add((min(a, int(b)), max(a, int(b))))
            if shown:
                lines.append("- characteristic spacings between neighbouring slots (normalised):")
                for a, b in sorted(shown):
                    ca, ia = self.slot_index[a]
                    cb, ib = self.slot_index[b]
                    lines.append(
                        f"    {ca}[{ia}] - {cb}[{ib}] : {self.edge_mean[a, b]:.3f} "
                        f"(+/- {self.edge_sigma[a, b]:.3f})"
                    )
        return "\n".join(lines)

    # -- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "category": self.category,
                    "n_shots": self.n_shots,
                    "classes": {c: cp.to_dict() for c, cp in self.classes.items()},
                    "slot_index": self.slot_index,
                    "edge_mean": None if self.edge_mean is None else self.edge_mean.round(4).tolist(),
                    "edge_sigma": None if self.edge_sigma is None else self.edge_sigma.round(4).tolist(),
                    "source_uids": self.source_uids,
                },
                indent=2,
            )
        )

    @staticmethod
    def load(path: str | Path) -> "NormalityGraph":
        d = json.loads(Path(path).read_text())
        return NormalityGraph(
            category=d["category"],
            n_shots=int(d["n_shots"]),
            classes={c: ClassPrior.from_dict(v) for c, v in d["classes"].items()},
            slot_index=[tuple(x) for x in d.get("slot_index", [])],
            edge_mean=None if d.get("edge_mean") is None else np.array(d["edge_mean"], dtype=float),
            edge_sigma=None if d.get("edge_sigma") is None else np.array(d["edge_sigma"], dtype=float),
            source_uids=d.get("source_uids", []),
        )


# -- helpers -------------------------------------------------------------


def _points_by_class(ann: ImageAnnotation) -> Dict[str, np.ndarray]:
    out: Dict[str, List[List[float]]] = {}
    for k in ann.keypoints:
        out.setdefault(k.cls, []).append([k.x, k.y])
    return {c: np.array(v, dtype=float).reshape(-1, 2) for c, v in out.items()}


def _groups_for(ann: ImageAnnotation, cls: str, m: int) -> List[Optional[str]]:
    g = [k.group for k in ann.keypoints if k.cls == cls]
    if len(g) < m:
        g = g + [None] * (m - len(g))
    return g[:m]
