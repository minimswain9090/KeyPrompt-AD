"""Turning a language-model verdict into a number you can plot an ROC curve on.

A VLM naturally emits a categorical answer ("NOT OK") plus a list of points.
Categorical answers give you a single operating point and nothing else, which
makes them awkward to compare against unsupervised baselines that report
AUROC. The scorer closes that gap: it audits the returned points against the
normality graph and produces a continuous deviation score.

Four terms are combined:

* missing   - prior slots with no nearby detection
* extra     - detections not explained by any slot
* displace  - offset of matched slots, in units of the learned tolerance
* edge      - violation of the learned pairwise spacings

The model's own verdict enters as a fifth, deliberately small term. The
geometry does the discriminating; the verdict mostly breaks ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ScoringConfig
from ..prior.geometry import align_by_class, apply_transform, match_points
from ..prior.graph import NormalityGraph


@dataclass
class ScoreBreakdown:
    score: float
    missing: float
    extra: float
    displacement: float
    edge: float
    vlm: float
    n_missing: int
    n_extra: int
    missing_points: List[Tuple[float, float]] = field(default_factory=list)
    extra_points: List[Tuple[float, float]] = field(default_factory=list)
    displaced_points: List[Tuple[float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def defect_points(self) -> List[Tuple[float, float]]:
        """Points to overlay on the image as the localisation output."""
        return self.missing_points + self.extra_points + self.displaced_points

    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 5),
            "terms": {
                "missing": round(self.missing, 5),
                "extra": round(self.extra, 5),
                "displacement": round(self.displacement, 5),
                "edge": round(self.edge, 5),
                "vlm": round(self.vlm, 5),
            },
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "missing_points": [[round(x, 4), round(y, 4)] for x, y in self.missing_points],
            "extra_points": [[round(x, 4), round(y, 4)] for x, y in self.extra_points],
            "displaced_points": [[round(x, 4), round(y, 4)] for x, y in self.displaced_points],
            "notes": self.notes,
        }


def score_detection(
    detected: Dict[str, List[Tuple[float, float]]],
    graph: NormalityGraph,
    cfg: ScoringConfig,
    vlm_verdict: Optional[str] = None,
    vlm_confidence: Optional[float] = None,
) -> ScoreBreakdown:
    """Compare a set of detected component positions against the prior."""
    pred = {c: np.asarray(v, dtype=float).reshape(-1, 2) for c, v in detected.items()}
    prior = graph.slots_by_class()

    # Global pose correction so that a shifted jig is not read as a defect.
    fit = align_by_class(pred, prior, radius=cfg.match_radius * 1.5)
    if fit is not None:
        R, s, t = fit
        pred = {c: apply_transform(p, R, s, t) for c, p in pred.items()}

    total_slots = max(graph.total_slots(), 1)
    n_missing = n_extra = 0
    disp_terms: List[float] = []
    missing_pts: List[Tuple[float, float]] = []
    extra_pts: List[Tuple[float, float]] = []
    displaced_pts: List[Tuple[float, float]] = []
    notes: List[str] = []

    matched_coords: Dict[Tuple[str, int], np.ndarray] = {}

    for cls, cp in graph.classes.items():
        pts = pred.get(cls, np.zeros((0, 2)))
        pairs, un_pred, un_prior = match_points(pts, cp.slots, cfg.match_radius)

        for pi, si in pairs:
            d = float(np.linalg.norm(pts[pi] - cp.slots[si]))
            z = min(d / max(cp.slot_sigma[si], 1e-6), cfg.sigma_clip)
            disp_terms.append(z / cfg.sigma_clip)
            matched_coords[(cls, si)] = pts[pi]
            if z > 3.0:
                displaced_pts.append((float(pts[pi][0]), float(pts[pi][1])))
                notes.append(f"{cls}[{si}] displaced by {d:.3f} ({z:.1f} sigma)")

        for si in un_prior:
            n_missing += 1
            missing_pts.append((float(cp.slots[si][0]), float(cp.slots[si][1])))
            notes.append(f"{cls}[{si}] not observed at expected location")

        for pi in un_pred:
            n_extra += 1
            extra_pts.append((float(pts[pi][0]), float(pts[pi][1])))
            notes.append(f"unexpected {cls} at ({pts[pi][0]:.3f}, {pts[pi][1]:.3f})")

    missing_term = n_missing / total_slots
    extra_term = n_extra / total_slots
    disp_term = float(np.mean(disp_terms)) if disp_terms else 0.0
    edge_term = _edge_violation(matched_coords, graph, cfg)

    vlm_term = 0.0
    if vlm_confidence is not None:
        vlm_term = float(np.clip(vlm_confidence, 0.0, 1.0))
    elif vlm_verdict is not None:
        vlm_term = 1.0 if str(vlm_verdict).strip().upper().startswith("NOT") else 0.0

    raw = (
        cfg.w_missing * missing_term
        + cfg.w_extra * extra_term
        + cfg.w_displacement * disp_term
        + cfg.w_edge * edge_term
        + cfg.w_vlm * vlm_term
    )
    denom = cfg.w_missing + cfg.w_extra + cfg.w_displacement + cfg.w_edge + cfg.w_vlm
    score = float(raw / denom) if denom > 0 else 0.0

    return ScoreBreakdown(
        score=score,
        missing=missing_term,
        extra=extra_term,
        displacement=disp_term,
        edge=edge_term,
        vlm=vlm_term,
        n_missing=n_missing,
        n_extra=n_extra,
        missing_points=missing_pts,
        extra_points=extra_pts,
        displaced_points=displaced_pts,
        notes=notes,
    )


def _edge_violation(
    matched: Dict[Tuple[str, int], np.ndarray],
    graph: NormalityGraph,
    cfg: ScoringConfig,
) -> float:
    """Mean normalised deviation of observed spacings from learned spacings."""
    if graph.edge_mean is None or len(matched) < 2 or not graph.slot_index:
        return 0.0
    keys = [k for k in matched if k in graph.slot_index]
    if len(keys) < 2:
        return 0.0

    zs: List[float] = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia = graph.slot_index.index(keys[a])
            ib = graph.slot_index.index(keys[b])
            mu = graph.edge_mean[ia, ib]
            sd = max(graph.edge_sigma[ia, ib], 1e-6)
            if mu <= 0:
                continue
            d = float(np.linalg.norm(matched[keys[a]] - matched[keys[b]]))
            zs.append(min(abs(d - mu) / sd, cfg.sigma_clip) / cfg.sigma_clip)
    return float(np.mean(zs)) if zs else 0.0
