"""Geometric primitives shared by the prior builder and the scorer.

Two operations carry most of the weight:

1. A similarity fit (rotation, uniform scale, translation) between two labelled
   point sets. Parts are rarely presented in a perfectly repeatable pose, and
   without alignment a two-pixel jig shift would masquerade as a defect.

2. A gated one-to-one assignment between predicted points and prior slots.
   Greedy nearest-neighbour matching produces order-dependent results, so the
   assignment is solved optimally and then gated by a radius, which lets points
   remain unmatched rather than forcing a bad pairing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def similarity_transform(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Umeyama's closed-form solution. Returns (R, s, t) such that
    ``s * src @ R.T + t`` approximates ``dst``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be matching (N, 2) arrays")
    n = src.shape[0]
    if n == 0:
        return np.eye(2), 1.0, np.zeros(2)
    if n == 1:
        return np.eye(2), 1.0, dst[0] - src[0]

    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return R, s, t


def apply_transform(pts: np.ndarray, R: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    return s * (pts @ R.T) + t


def match_points(
    pred: np.ndarray,
    prior: np.ndarray,
    radius: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Optimal one-to-one assignment of predictions to prior slots.

    Returns ``(pairs, unmatched_pred, unmatched_prior)`` where ``pairs`` holds
    (pred_index, prior_index) tuples whose separation is within ``radius``.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    prior = np.asarray(prior, dtype=float).reshape(-1, 2)
    if len(pred) == 0:
        return [], [], list(range(len(prior)))
    if len(prior) == 0:
        return [], list(range(len(pred))), []

    cost = np.linalg.norm(pred[:, None, :] - prior[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(cost)

    pairs: List[Tuple[int, int]] = []
    used_pred, used_prior = set(), set()
    for i, j in zip(ri, ci):
        if cost[i, j] <= radius:
            pairs.append((int(i), int(j)))
            used_pred.add(int(i))
            used_prior.add(int(j))
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    unmatched_prior = [j for j in range(len(prior)) if j not in used_prior]
    return pairs, unmatched_pred, unmatched_prior


def align_by_class(
    pred_by_class: dict[str, np.ndarray],
    prior_by_class: dict[str, np.ndarray],
    radius: float,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Estimate a global pose correction from confidently matched points.

    A first pass matches without alignment to find a consensus set; the
    transform is then fitted on that set only, so missing or spurious points do
    not drag the pose estimate.
    """
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    for cls, prior_pts in prior_by_class.items():
        pred_pts = pred_by_class.get(cls)
        if pred_pts is None or len(pred_pts) == 0 or len(prior_pts) == 0:
            continue
        pairs, _, _ = match_points(pred_pts, prior_pts, radius)
        for pi, qi in pairs:
            src_pts.append(pred_pts[pi])
            dst_pts.append(prior_pts[qi])
    if len(src_pts) < 2:
        return None
    return similarity_transform(np.array(src_pts), np.array(dst_pts))


def pairwise_distances(pts: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
