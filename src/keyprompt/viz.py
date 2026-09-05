"""Qualitative output.

The classical post-processing step from the original design: take the
coordinates the model returned, project them back onto the full-resolution
image, and draw them. Cheap, deterministic, and the figure most reviewers look
at first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "missing": (220, 40, 40),
    "extra": (250, 160, 30),
    "displaced": (70, 110, 230),
    "ok": (40, 190, 110),
}


def overlay(
    image: Image.Image,
    breakdown: Optional[Dict] = None,
    extra_points: Sequence[Tuple[float, float]] = (),
    gt_mask: Optional[np.ndarray] = None,
    radius_frac: float = 0.022,
) -> Image.Image:
    """Draw predicted defect coordinates, optionally over the ground-truth mask."""
    out = image.convert("RGB")
    W, H = out.size
    r = max(int(radius_frac * max(W, H)), 6)

    if gt_mask is not None:
        m = Image.fromarray((gt_mask.astype(np.uint8) * 255)).resize((W, H))
        tint = Image.new("RGB", (W, H), (60, 200, 90))
        out = Image.composite(Image.blend(out, tint, 0.35), out, m)

    draw = ImageDraw.Draw(out, "RGBA")

    def ring(x: float, y: float, color: Tuple[int, int, int], label: str = "") -> None:
        cx, cy = x * W, y * H
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (110,), outline=color, width=3)
        if label:
            draw.text((cx + r + 4, cy - r), label, fill=color)

    if breakdown:
        for x, y in breakdown.get("missing_points", []):
            ring(x, y, COLORS["missing"], "missing")
        for x, y in breakdown.get("extra_points", []):
            ring(x, y, COLORS["extra"], "extra")
        for x, y in breakdown.get("displaced_points", []):
            ring(x, y, COLORS["displaced"], "shifted")
    for x, y in extra_points:
        ring(x, y, COLORS["missing"])

    return out


def save_grid(
    images: List[Image.Image],
    out_path: str | Path,
    cols: int = 4,
    cell: int = 320,
) -> Path:
    """Contact sheet of qualitative results, for the figures in the paper."""
    if not images:
        raise ValueError("no images to write")
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, im in enumerate(images):
        t = im.copy()
        t.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x = (i % cols) * cell + (cell - t.width) // 2
        y = (i // cols) * cell + (cell - t.height) // 2
        sheet.paste(t, (x, y))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(p)
    return p
