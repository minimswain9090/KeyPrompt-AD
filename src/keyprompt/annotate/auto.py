"""Deriving reference annotations without hand-clicking every point.

Manual annotation is the weakest link in the pipeline, not because it is slow
but because four hand-placed points are a poor sample from which to estimate a
positional tolerance. The consensus route below is cheaper *and* statistically
better: propose points on many normal training images, align them into a common
frame, and keep only the clusters that recur across most of them.

Two proposers are provided.

``propose_blobs``
    Classical connected-component detection on a thresholded image. No model,
    no network, deterministic. Works well when components are visually distinct
    from the background (pushpins in a tray, screws on a light surface) and
    poorly otherwise. Reach for this first.

``propose_vlm``
    Asks the vision-language model to locate components from the text normality
    statement alone, with no coordinates supplied. Fully hands-off, and the
    obvious thing to try -- but note the failure mode: the prior is built from
    the model's own detections and then used to grade that same model. Errors
    are absorbed rather than exposed. Report results from this route separately
    from the manually-annotated ones; do not silently mix them.

Neither proposer knows which cluster is which component class when several
classes are present. For multi-class categories (screw_bag, juice_bottle) the
practical route is proposal followed by a quick human pass to assign labels,
which is still far less work than placing every point by hand.

Nothing here ever reads ``ground_truth/``. Those masks exist only for anomalous
test images; using them to build the normality prior would be test-set leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..annotate.schema import ImageAnnotation, Keypoint
from ..prior.geometry import align_by_class, apply_transform, match_points

Proposal = Tuple[str, float, float]  # (class, x, y) in normalised coordinates
Proposer = Callable[[Image.Image], List[Proposal]]


# -- proposers -----------------------------------------------------------


def propose_blobs(
    image: Image.Image,
    cls: str = "component",
    min_area_frac: float = 2e-4,
    max_area_frac: float = 5e-2,
    invert: bool = False,
    threshold: Optional[float] = None,
    max_points: int = 64,
) -> List[Proposal]:
    """Connected-component proposal on a thresholded grayscale image.

    ``threshold`` defaults to Otsu's method. Set ``invert`` when components are
    darker than the background. Area bounds are expressed as a fraction of the
    image so the same settings survive a change of resolution.
    """
    from scipy import ndimage

    g = np.asarray(image.convert("L"), dtype=float) / 255.0
    thr = _otsu(g) if threshold is None else threshold
    mask = (g < thr) if invert else (g > thr)

    # Close small gaps so a specular highlight does not split one component.
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

    labels, n = ndimage.label(mask)
    if n == 0:
        return []

    H, W = g.shape
    area = H * W
    lo, hi = min_area_frac * area, max_area_frac * area
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if lo <= s <= hi]
    if not keep:
        return []

    centroids = ndimage.center_of_mass(mask, labels, index=keep)
    ranked = sorted(zip(keep, centroids), key=lambda kv: -sizes[kv[0] - 1])[:max_points]
    return [(cls, float(c[1] / W), float(c[0] / H)) for _, c in ranked]


def propose_vlm(
    image: Image.Image,
    provider,
    category_cfg,
    image_cfg,
) -> List[Proposal]:
    """Ask the VLM to locate components from the text description alone.

    No coordinates are supplied, so this does not depend on any prior existing
    yet. It is the bootstrap step for a fully hands-off run.
    """
    from ..prompting.schema import parse_response
    from ..providers.base import prepare_image

    classes = category_cfg.component_classes or ["component"]
    prompt = f"""
This image shows a correct, non-defective example of '{category_cfg.name}'.

{category_cfg.normality_statement.strip()}

Locate every instance of these component types: {", ".join(classes)}.

Report the centre of each one in normalised coordinates, where x runs from 0 at
the left edge to 1 at the right edge and y from 0 at the top to 1 at the bottom.
Do not omit any instance, and do not invent instances that are not visible.

Reply with a single JSON object and nothing else:
{{"verdict": "OK", "confidence": 0.0,
  "detected": [{{"cls": "<one of: {', '.join(classes)}>", "x": <float>, "y": <float>}}],
  "missing": [], "reasoning": ""}}
""".strip()

    resp = provider.generate([prompt, prepare_image(image, image_cfg)])
    if resp.error:
        return []
    try:
        parsed = parse_response(resp.text)
    except Exception:  # noqa: BLE001 - a bad proposal is dropped, not fatal
        return []
    return [(d.cls, d.x, d.y) for d in parsed.detected]


# -- consensus -----------------------------------------------------------


@dataclass
class ConsensusReport:
    n_images: int
    n_slots: int
    per_class: Dict[str, int]
    dropped_clusters: int
    mean_support: float
    mean_spread: float

    def describe(self) -> str:
        return (
            f"consensus over {self.n_images} normal images\n"
            f"  surviving slots : {self.n_slots} {self.per_class}\n"
            f"  dropped clusters: {self.dropped_clusters} (below support threshold)\n"
            f"  mean support    : {self.mean_support:.2f} of images\n"
            f"  mean spread     : {self.mean_spread:.4f} (normalised units)"
        )


def consensus_annotations(
    images: Sequence[Image.Image],
    proposer: Proposer,
    min_support: float = 0.6,
    cluster_radius: float = 0.05,
    align: bool = True,
) -> Tuple[List[ImageAnnotation], ConsensusReport]:
    """Turn noisy per-image proposals into clean reference annotations.

    A cluster of points that recurs at the same place across most normal images
    is a real component slot. One that appears in a handful is a detection
    artefact. That distinction is the whole filter, and it is what makes 30
    automatic proposals more trustworthy than 4 careful manual clicks.

    Returns annotations in the same format the manual tool produces, so the
    downstream graph builder is unchanged.
    """
    if not images:
        raise ValueError("no images supplied")

    raw: List[Dict[str, np.ndarray]] = []
    for img in images:
        by_cls: Dict[str, List[List[float]]] = {}
        for cls, x, y in proposer(img):
            by_cls.setdefault(cls, []).append([x, y])
        raw.append({c: np.array(v, dtype=float).reshape(-1, 2) for c, v in by_cls.items()})

    # Anchor on the image with the median number of proposals: an outlier frame
    # where the detector over- or under-fired should not define the frame.
    totals = [sum(len(p) for p in r.values()) for r in raw]
    anchor_idx = int(np.argsort(totals)[len(totals) // 2])
    anchor = raw[anchor_idx]

    aligned: List[Dict[str, np.ndarray]] = []
    for i, r in enumerate(raw):
        if not align or i == anchor_idx:
            aligned.append(r)
            continue
        fit = align_by_class(r, anchor, radius=cluster_radius * 2.0)
        if fit is None:
            aligned.append(r)
        else:
            R, s, t = fit
            aligned.append({c: apply_transform(p, R, s, t) for c, p in r.items()})

    classes = sorted({c for a in aligned for c in a})
    slots: Dict[str, np.ndarray] = {}
    supports: Dict[str, np.ndarray] = {}
    spreads: List[float] = []
    dropped = 0

    for cls in classes:
        pooled, owners = [], []
        for i, a in enumerate(aligned):
            for p in a.get(cls, np.zeros((0, 2))):
                pooled.append(p)
                owners.append(i)
        if not pooled:
            continue

        centres, members = _cluster(np.array(pooled), cluster_radius)
        keep_centres, keep_support = [], []
        for c, idxs in zip(centres, members):
            support = len({owners[j] for j in idxs}) / len(images)
            if support >= min_support:
                keep_centres.append(c)
                keep_support.append(support)
                pts = np.array(pooled)[idxs]
                spreads.append(float(pts.std(axis=0).mean()))
            else:
                dropped += 1
        if keep_centres:
            slots[cls] = np.array(keep_centres)
            supports[cls] = np.array(keep_support)

    # Emit one annotation per source image, keeping only proposals that joined a
    # surviving cluster, expressed back in that image's own coordinates.
    annotations: List[ImageAnnotation] = []
    for i, (img, r) in enumerate(zip(images, raw)):
        kps: List[Keypoint] = []
        for cls, centres in slots.items():
            pts = aligned[i].get(cls, np.zeros((0, 2)))
            if len(pts) == 0:
                continue
            pairs, _, _ = match_points(pts, centres, cluster_radius)
            for pi, si in pairs:
                orig = r[cls][pi]  # untransformed, matching this image
                kps.append(
                    Keypoint(cls=cls, x=float(orig[0]), y=float(orig[1]), group=f"{cls}_{si}")
                )
        annotations.append(
            ImageAnnotation(
                image_uid=f"auto/{i}",
                category="auto",
                width=img.width,
                height=img.height,
                keypoints=kps,
            )
        )

    report = ConsensusReport(
        n_images=len(images),
        n_slots=sum(len(v) for v in slots.values()),
        per_class={c: len(v) for c, v in slots.items()},
        dropped_clusters=dropped,
        mean_support=float(np.mean([s for v in supports.values() for s in v])) if supports else 0.0,
        mean_spread=float(np.mean(spreads)) if spreads else 0.0,
    )
    return annotations, report


# -- helpers -------------------------------------------------------------


def _cluster(points: np.ndarray, radius: float) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Greedy agglomerative clustering, densest seed first.

    Simple by design: the point sets are small and the geometry is already
    aligned, so a heavier algorithm buys nothing but another dependency.
    """
    n = len(points)
    if n == 0:
        return [], []
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    density = (d <= radius).sum(axis=1)
    order = np.argsort(-density)

    unassigned = set(range(n))
    centres: List[np.ndarray] = []
    members: List[List[int]] = []
    for seed in order:
        if seed not in unassigned:
            continue
        group = [j for j in unassigned if d[seed, j] <= radius]
        unassigned -= set(group)
        centres.append(points[group].mean(axis=0))
        members.append(group)
    return centres, members


def _otsu(g: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(g.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b = np.nan_to_num(sigma_b)

    # On a cleanly separated image, between-class variance is flat across the
    # whole empty valley between the two modes. Taking argmax would put the
    # threshold on the first tied bin, i.e. hard against the darker cluster,
    # where any noise splits those pixels arbitrarily. Take the middle of the
    # tied range instead so the threshold sits in the gap.
    best = sigma_b.max()
    if best <= 0:
        return 0.5
    tied = np.flatnonzero(sigma_b >= best - 1e-12)
    return float(mids[int(round((tied[0] + tied[-1]) / 2.0))])
