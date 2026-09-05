#!/usr/bin/env python3
"""Standalone visualiser: render predicted defect coordinates over the images.

Drop this file in the repository root and run:

    python visualize.py --config configs/pushpins_only.yaml --category pushpins

Writes individual overlays plus a contact sheet to
runs/<output_root>/<category>/figures/.

Colour key:
    red    a slot the prior expects but nothing was detected there
    orange a detection that matches no slot
    blue   a matched component displaced beyond three sigma
    green  tint marks the annotated ground-truth region, for comparison

This duplicates the `keyprompt visualize` subcommand so it can be used without
reinstalling the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from keyprompt.config import RunConfig
from keyprompt.data.loco import CATEGORIES, LocoCategory
from keyprompt.viz import overlay, save_grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pushpins_only.yaml")
    ap.add_argument("--category", default="pushpins", choices=CATEGORIES)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--only", choices=["all", "anomalous", "normal", "errors"],
                    default="anomalous")
    ap.add_argument("--no-ground-truth", dest="show_gt", action="store_false", default=True)
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    run_dir = Path(cfg.output_root) / args.category
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        sys.exit(f"no predictions at {pred_file}; run the pipeline first")

    rows = [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]
    if args.only == "anomalous":
        rows = [r for r in rows if r["label"] == 1]
    elif args.only == "normal":
        rows = [r for r in rows if r["label"] == 0]
    elif args.only == "errors":
        rows = [r for r in rows if not r.get("parse_ok", True) or r.get("error")]

    # Highest score first: the most confident calls deserve the closest look,
    # whether they are right or wrong.
    rows.sort(key=lambda r: -r.get("score", 0.0))
    rows = rows[: args.limit]
    if not rows:
        sys.exit("nothing to visualise with those filters")

    cat = LocoCategory.open(cfg.dataset_root, args.category)
    by_uid = {s.uid: s for s in cat.test(limit_per_subset=cfg.limit_per_split)}

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for r in rows:
        sample = by_uid.get(r["uid"])
        if sample is None:
            continue
        img = overlay(
            sample.load_image(),
            breakdown=r.get("breakdown"),
            extra_points=[tuple(p) for p in r.get("defect_points", [])],
            gt_mask=sample.load_gt_union() if args.show_gt else None,
        )
        img.save(out_dir / f"{r['subset']}_{sample.stem}_score{r['score']:.3f}.png")
        panels.append(img)

    if not panels:
        sys.exit("no matching images found on disk")

    sheet = save_grid(panels, out_dir / "contact_sheet.png", cols=args.cols)
    print(f"wrote {len(panels)} overlays to {out_dir}")
    print(f"contact sheet: {sheet}")
    print("red = missing, orange = unexpected extra, blue = displaced beyond tolerance")
    if args.show_gt:
        print("green tint = annotated ground-truth defect region")


if __name__ == "__main__":
    main()
