"""Metrics.

Three families are reported, because a paper that only reports AUROC invites
the obvious question of whether the system is usable on a line.

Detection
    Image-level AUROC and average precision, plus the best achievable F1 and
    the accuracy at that operating point. AUROC is also broken out for the
    logical and structural test subsets separately, since MVTec LOCO
    deliberately separates the two and the method is aimed squarely at the
    logical half.

Localisation
    The official LOCO localisation metric (saturated per-region overlap)
    assumes dense masks; this system emits points, so a point-based analogue is
    used instead. Region recall asks what fraction of annotated defect regions
    contain at least one predicted point. Point precision asks what fraction of
    predicted points land inside some annotated region. Both are reported over
    anomalous test images only, and the choice is stated rather than smuggled
    in as if it were the official protocol.

Efficiency
    Latency percentiles and token counts, measured on uncached calls only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -- detection -----------------------------------------------------------


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUROC via the rank statistic, with correct handling of ties."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def best_f1(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    """Maximum F1 over all thresholds, and the threshold that achieves it."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    k = int(np.argmax(f1))
    return float(f1[k]), float(ss[k])


def classification_at(labels: Sequence[int], scores: Sequence[float], thr: float) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


# -- localisation --------------------------------------------------------


@dataclass
class LocalisationCase:
    points: List[Tuple[float, float]]   # normalised predicted defect points
    regions: List[np.ndarray]           # boolean GT masks, image resolution


def localisation_scores(cases: List[LocalisationCase], tolerance_px: int = 12) -> Dict[str, float]:
    """Point-based localisation.

    A predicted point counts as hitting a region if it falls inside the mask, or
    within ``tolerance_px`` of it. The tolerance matters for "missing component"
    defects, where the annotated region marks an empty slot and the natural
    prediction sits at its centre but need not land exactly inside.
    """
    hit_regions = total_regions = 0
    hit_points = total_points = 0
    images_localised = images_with_regions = 0

    for case in cases:
        if not case.regions:
            continue
        images_with_regions += 1
        H, W = case.regions[0].shape
        px = [(int(round(x * W)), int(round(y * H))) for x, y in case.points]
        dil = [_dilate(r, tolerance_px) for r in case.regions]

        any_hit = False
        for r in dil:
            total_regions += 1
            if any(_inside(r, u, v) for u, v in px):
                hit_regions += 1
                any_hit = True
        for u, v in px:
            total_points += 1
            if any(_inside(r, u, v) for r in dil):
                hit_points += 1
        if any_hit:
            images_localised += 1

    return {
        "region_recall": hit_regions / total_regions if total_regions else float("nan"),
        "point_precision": hit_points / total_points if total_points else float("nan"),
        "image_hit_rate": images_localised / images_with_regions if images_with_regions else float("nan"),
        "n_regions": total_regions,
        "n_points": total_points,
    }


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

        st = iterate_structure(generate_binary_structure(2, 1), r)
        return binary_dilation(mask, structure=st)
    except ImportError:
        return mask


def _inside(mask: np.ndarray, u: int, v: int) -> bool:
    h, w = mask.shape
    return 0 <= v < h and 0 <= u < w and bool(mask[v, u])


# -- aggregation ---------------------------------------------------------


def summarise(
    labels: Sequence[int],
    scores: Sequence[float],
    subsets: Sequence[str],
    latencies: Sequence[float],
    prompt_tokens: Sequence[Optional[int]] = (),
    output_tokens: Sequence[Optional[int]] = (),
    parse_ok: Sequence[bool] = (),
) -> Dict[str, object]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    sub = np.asarray(subsets)

    f1, thr = best_f1(y, s)
    out: Dict[str, object] = {
        "n": int(len(y)),
        "n_normal": int((y == 0).sum()),
        "n_anomalous": int((y == 1).sum()),
        "auroc": roc_auc(y, s),
        "average_precision": average_precision(y, s),
        "f1_max": f1,
        "operating_point": classification_at(y, s, thr),
    }

    normal_mask = sub == "good"
    for name in ("logical_anomalies", "structural_anomalies"):
        m = normal_mask | (sub == name)
        if (sub == name).sum() > 0:
            out[f"auroc_{name.replace('_anomalies', '')}"] = roc_auc(y[m], s[m])

    lat = np.asarray([x for x in latencies if x and x > 0], dtype=float)
    if lat.size:
        out["latency"] = {
            "mean_s": float(lat.mean()),
            "p50_s": float(np.percentile(lat, 50)),
            "p95_s": float(np.percentile(lat, 95)),
            "max_s": float(lat.max()),
        }
    pt = [x for x in prompt_tokens if x]
    ot = [x for x in output_tokens if x]
    if pt or ot:
        out["tokens"] = {
            "prompt_mean": float(np.mean(pt)) if pt else None,
            "output_mean": float(np.mean(ot)) if ot else None,
        }
    if len(parse_ok):
        out["parse_success_rate"] = float(np.mean([1.0 if x else 0.0 for x in parse_ok]))
    return out
