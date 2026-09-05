#!/usr/bin/env python3
"""Render the keypoint and grouping figure from your own annotations.

Two panels, both drawn from real files rather than illustration:

  (a) the keypoints as annotated on one reference tray, coloured by group
  (b) the normality graph fitted from all of them: canonical slots, their
      learned tolerances, and the within-group spacing edges

Run from the repository root:

    python make_annotation_figure.py --config configs/pushpins_only.yaml

Writes fig_annotation_pushpins.pdf next to the paper sources. Nothing here is
invented; if the figure looks wrong, the prior is wrong, which is itself worth
knowing before a full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from keyprompt.annotate.schema import load_annotation_set
from keyprompt.config import RunConfig
from keyprompt.data.loco import LocoCategory
from keyprompt.prior.graph import NormalityGraph

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#bfef45", "#fabed4", "#469990"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pushpins_only.yaml")
    ap.add_argument("--category", default="pushpins")
    ap.add_argument("--out", default="fig_annotation_pushpins.pdf")
    ap.add_argument("--max-edges", type=int, default=2,
                    help="nearest-neighbour spacing edges drawn per slot")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    ann_dir = Path(cfg.annotations_root) / args.category
    prior_path = Path(cfg.output_root) / args.category / "prior.json"

    anns = load_annotation_set(ann_dir)
    if not anns:
        sys.exit(f"no annotations in {ann_dir}; run bootstrap or annotate first")
    if not prior_path.exists():
        sys.exit(f"no prior at {prior_path}; run build-prior first")

    graph = NormalityGraph.load(prior_path)
    cat = LocoCategory.open(cfg.dataset_root, args.category)
    by_stem = {s.stem: s for s in cat.train_normal()}

    # Use the most complete annotation, which is the one the prior anchored on.
    anns.sort(key=lambda a: -len(a.keypoints))
    ref = anns[0]
    stem = ref.image_uid.split("/")[-1]
    if stem not in by_stem:
        sys.exit(f"reference image {stem} not found under the dataset root")
    img = by_stem[stem].load_image()
    W, H = img.size

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0))

    # -- panel (a): the annotation itself ----------------------------
    ax = axes[0]
    ax.imshow(img)
    groups = sorted({k.group for k in ref.keypoints if k.group},
                    key=lambda g: str(g))
    gidx = {g: i for i, g in enumerate(groups)}
    for k in ref.keypoints:
        colour = PALETTE[gidx.get(k.group, 0) % len(PALETTE)] if k.group else PALETTE[0]
        ax.add_patch(Circle((k.x * W, k.y * H), max(W, H) * 0.011,
                            fc=colour, ec="white", lw=1.2, zorder=3))
    ax.set_title(f"(a) Keypoint annotation on one reference tray: "
                 f"{len(ref.keypoints)} components, coloured by group",
                 fontsize=9)
    ax.axis("off")

    # -- panel (b): the fitted graph ---------------------------------
    ax = axes[1]
    ax.imshow(img, alpha=0.45)
    cls = sorted(graph.classes)[0]
    cp = graph.classes[cls]
    slots = cp.slots

    if graph.edge_mean is not None and len(graph.slot_index) >= 2:
        drawn = set()
        for a in range(len(graph.slot_index)):
            row = graph.edge_mean[a].copy()
            row[a] = np.inf
            row[row <= 0] = np.inf
            for b in np.argsort(row)[: args.max_edges]:
                if not np.isfinite(row[b]):
                    continue
                pair = (min(a, int(b)), max(a, int(b)))
                if pair in drawn:
                    continue
                drawn.add(pair)
                p, q = slots[pair[0]], slots[pair[1]]
                ax.plot([p[0] * W, q[0] * W], [p[1] * H, q[1] * H],
                        color="#1b5e20", lw=1.0, ls=(0, (4, 3)), zorder=2)

    for i, (xy, sg) in enumerate(zip(slots, cp.slot_sigma)):
        cx, cy = xy[0] * W, xy[1] * H
        r = max(sg * max(W, H), max(W, H) * 0.004)
        ax.add_patch(Circle((cx, cy), r, fc="#1976d2", ec="#1976d2",
                            alpha=0.22, lw=1.0, zorder=3))
        ax.add_patch(Circle((cx, cy), max(W, H) * 0.008, fc="#0d47a1",
                            ec="white", lw=1.0, zorder=4))
        ax.annotate(str(i), (cx, cy), textcoords="offset points",
                    xytext=(7, 6), fontsize=6, color="#0d47a1", zorder=5)

    ax.set_title(f"(b) Normality graph from {graph.n_shots} references: "
                 f"{graph.total_slots()} slots, learned tolerances (shaded), "
                 f"spacing edges (dashed)", fontsize=9)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")
    print(f"  reference image : {stem}")
    print(f"  keypoints drawn : {len(ref.keypoints)}")
    print(f"  slots in prior  : {graph.total_slots()}")
    print(f"  mean tolerance  : {float(cp.slot_sigma.mean()):.4f} (normalised)")
    if float(cp.slot_sigma.mean()) < 0.02:
        print("\n  NOTE: that tolerance is very tight. For components that sit")
        print("  loosely inside a compartment it will make the displacement term")
        print("  fire on correct parts. See Section 5.3 of the manuscript.")


if __name__ == "__main__":
    main()
