#!/usr/bin/env python3
"""Generate the manuscript's *methodology* figures.

These illustrate how the method works. They are not experimental results and
must never be presented as such: every panel is drawn from a synthetic
configuration so that the mechanism is visible without a dataset. Result
figures come from make_paper_assets.py once the benchmark has been run.

    python make_method_figures.py --out paper/

Produces:
    fig_keypoint_grouping.pdf   annotation scheme: classes, groups, shape edges
    fig_pipeline.pdf            block diagram of the five stages
    fig_audit.pdf               the geometric audit on three defect modes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

NUT = "#1f77b4"
HOLE = "#d62728"
SLOT = "#8c8c8c"
OKC = "#2ca02c"


# -- Figure 1: the keypoint and grouping mechanism -----------------------


def fig_keypoint_grouping(out: Path) -> Path:
    """The annotation scheme, on a stylised bracket with two nuts and a hole.

    Left: what the annotator marks. Right: what is derived from it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    nuts = np.array([[0.18, 0.72], [0.74, 0.70]])
    hole = np.array([[0.47, 0.28]])

    # -- left: annotation --------------------------------------------
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((0.06, 0.14), 0.84, 0.70,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="#eceff1", ec="#90a4ae", lw=1.2))
    tri = np.vstack([nuts, hole])
    ax.add_patch(plt.Polygon(tri, closed=True, fill=False,
                             ec="#455a64", lw=1.4, ls=(0, (5, 3))))
    for i, p in enumerate(nuts):
        ax.add_patch(Circle(p, 0.045, fc=NUT, ec="white", lw=1.6, zorder=3))
        ax.annotate(f"nut[{i}]", p + np.array([0.0, 0.085]), ha="center",
                    fontsize=8, color=NUT)
    ax.add_patch(Circle(hole[0], 0.045, fc=HOLE, ec="white", lw=1.6, zorder=3))
    ax.annotate("hole[0]", hole[0] + np.array([0.0, -0.11]), ha="center",
                fontsize=8, color=HOLE)
    ax.annotate("group $g_1$", (0.47, 0.545), ha="center", fontsize=8.5,
                color="#455a64", style="italic")
    ax.set_title("(a) What the annotator marks\n"
                 "one click per component, tagged with a class and a group",
                 fontsize=9)

    # -- right: what is derived --------------------------------------
    ax = axes[1]
    ax.add_patch(FancyBboxPatch((0.06, 0.14), 0.84, 0.70,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="#eceff1", ec="#90a4ae", lw=1.2))
    for i, p in enumerate(np.vstack([nuts, hole])):
        col = NUT if i < 2 else HOLE
        ax.add_patch(Circle(p, 0.075, fc=col, ec="none", alpha=0.16, zorder=1))
        ax.add_patch(Circle(p, 0.075, fc="none", ec=col, lw=0.9,
                            ls=(0, (2, 2)), zorder=2))
        ax.add_patch(Circle(p, 0.030, fc=col, ec="white", lw=1.3, zorder=3))
    ax.annotate(r"$\sigma$ tolerance", nuts[0] + np.array([-0.02, 0.135]),
                ha="center", fontsize=7.5, color="#37474f")

    pairs = [(nuts[0], nuts[1], r"$E_{01}\pm\tau_{01}$", 0.055),
             (nuts[0], hole[0], r"$E_{02}$", -0.05),
             (nuts[1], hole[0], r"$E_{12}$", -0.05)]
    for a, b, lab, off in pairs:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="<->", color="#546e7a", lw=1.0))
        m = (a + b) / 2
        ax.annotate(lab, m + np.array([0.0, off]), ha="center", fontsize=7.5,
                    color="#37474f")
    ax.set_title("(b) What the normality graph stores\n"
                 "slot positions with tolerances, and pairwise spacings",
                 fontsize=9)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.05, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    p = out / "fig_keypoint_grouping.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# -- Figure 2: pipeline --------------------------------------------------


def fig_pipeline(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    stages = [
        ("$K$ reference\nimages", "#e3f2fd"),
        ("keypoint +\ngroup annotation", "#e3f2fd"),
        ("normality\ngraph $\\mathcal{G}$", "#fff8e1"),
        ("generated layout\nspecification", "#fff8e1"),
        ("VLM\n(one call)", "#f3e5f5"),
        ("geometric\naudit", "#e8f5e9"),
        ("score $s(x)$ +\ncoordinates", "#e8f5e9"),
    ]
    w, h, gap = 1.05, 0.72, 0.30
    for i, (label, colour) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.42), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.06",
                                    fc=colour, ec="#607d8b", lw=1.0))
        ax.annotate(label, (x + w / 2, 0.78), ha="center", va="center", fontsize=7.6)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 0.78),
                                         (x + w + gap - 0.03, 0.78),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color="#607d8b", lw=1.0))

    # query image enters at the model, from above
    qx = 4 * (w + gap)
    ax.add_patch(FancyBboxPatch((qx, 1.45), w, 0.42,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc="#fce4ec", ec="#607d8b", lw=1.0))
    ax.annotate("query image", (qx + w / 2, 1.66), ha="center", va="center", fontsize=7.6)
    ax.add_patch(FancyArrowPatch((qx + w / 2, 1.42), (qx + w / 2, 1.17),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="#607d8b", lw=1.0))

    # The graph is reused by the audit; that reuse is the core of the design.
    # Routed below the row so it crosses no box.
    gx = 2 * (w + gap) + w / 2
    ax_x = 5 * (w + gap) + w / 2
    ax.add_patch(FancyArrowPatch((gx, 0.40), (ax_x, 0.40),
                                 connectionstyle="arc3,rad=0.42",
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="#2e7d32", lw=1.2, ls=(0, (4, 2))))
    ax.annotate("the same graph both specifies the layout and verifies the answer",
                ((gx + ax_x) / 2, -0.46), ha="center", fontsize=7.4,
                color="#2e7d32", style="italic")

    ax.annotate("no gradient updates at any stage",
                (0.0, 1.66), fontsize=7.6, color="#546e7a", style="italic")
    ax.set_xlim(-0.15, 7 * (w + gap))
    ax.set_ylim(-0.62, 1.95)
    ax.axis("off")
    fig.tight_layout()
    p = out / "fig_pipeline.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# -- Figure 3: the audit on three defect modes ---------------------------


def fig_audit(out: Path) -> Path:
    """Illustrate the audit. Scores come from the real scoring code."""
    import sys
    sys.path.insert(0, "src")
    from keyprompt.annotate.schema import ImageAnnotation, Keypoint
    from keyprompt.config import ScoringConfig
    from keyprompt.pipeline.scoring import score_detection
    from keyprompt.prior.graph import NormalityGraph

    grid = [(0.12 + 0.19 * c, 0.22 + 0.30 * r) for r in range(3) for c in range(5)]
    rng = np.random.default_rng(0)
    anns = [
        ImageAnnotation(f"r{k}", "synthetic", 1000, 600,
                        [Keypoint("pushpin", x + rng.normal(0, 0.004),
                                  y + rng.normal(0, 0.004)) for x, y in grid])
        for k in range(4)
    ]
    graph = NormalityGraph.build(anns, "synthetic")
    cfg = ScoringConfig()

    cases = [
        ("correct part", list(grid), "OK", 0.0),
        ("missing component", [p for i, p in enumerate(grid) if i != 7], "NOT OK", 0.9),
        ("displaced component",
         [(x + (0.09 if i == 3 else 0.0), y) for i, (x, y) in enumerate(grid)], "NOT OK", 0.8),
        ("global pose shift", [(x + 0.03, y - 0.02) for x, y in grid], "OK", 0.0),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.7))
    for ax, (title, pts, verdict, conf) in zip(axes, cases):
        b = score_detection({"pushpin": pts}, graph, cfg, verdict, conf)
        for s in graph.classes["pushpin"].slots:
            ax.add_patch(Circle(s, 0.030, fc="none", ec=SLOT, lw=0.8, ls=(0, (2, 2))))
        for p in pts:
            ax.add_patch(Circle(p, 0.019, fc="#37474f", ec="none"))
        for p in b.missing_points:
            ax.add_patch(Circle(p, 0.045, fc=HOLE, ec=HOLE, alpha=0.28, lw=1.5))
        for p in b.displaced_points:
            ax.add_patch(Circle(p, 0.045, fc="#1f77b4", ec="#1f77b4", alpha=0.28, lw=1.5))
        colour = OKC if b.score < 0.05 else "#c62828"
        ax.set_title(f"{title}\n$s(x)={b.score:.3f}$", fontsize=8.5, color=colour)
        ax.set_xlim(0, 1)
        ax.set_ylim(1.0, 0.0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    p = out / "fig_audit.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_keypoint_grouping, fig_pipeline, fig_audit):
        print(f"  wrote {fn(args.out)}")
    print("\nThese are methodology illustrations drawn from synthetic "
          "configurations. They are not experimental results.")


if __name__ == "__main__":
    main()
