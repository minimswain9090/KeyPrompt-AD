#!/usr/bin/env python3
"""Run the ablations the paper needs, one variable at a time.

Four sweeps, each holding everything else fixed:

  shots       1, 2, 4, 8 reference images. The headline claim is that four is
              enough, so the curve has to be shown rather than asserted.
  resolution  256 to 768 px. Establishes the accuracy/latency trade-off and
              tells a deployment reader what to set.
  terms       Scoring-weight ablation. Zeroing one term at a time shows which
              part of the geometric audit is doing the work, and in particular
              whether the edge term earns its place.
  model       The same prompts against a different backend, which separates
              "the method works" from "one model happens to be good at this".

Each variant writes to its own run directory, so nothing is overwritten and a
sweep can be resumed after a rate-limit stall.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def write_variant(base: dict, name: str, patch: dict) -> Path:
    cfg = copy.deepcopy(base)
    for dotted, value in patch.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    cfg["output_root"] = f"runs/sweep/{name}"
    out = REPO / "configs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_variant(config: Path, categories: list[str], skip_run: bool) -> None:
    for cat in categories:
        # The prior is refitted per variant because it depends on the alignment
        # and tolerance settings; annotations themselves are shared and are read
        # from the stable annotations_root.
        stages = ["build-prior", "run", "evaluate"] if not skip_run else ["evaluate"]
        for stage in stages:
            cmd = [sys.executable, "-m", "keyprompt.cli", "--config", str(config), stage,
                   "--category", cat]
            print("$", " ".join(cmd))
            subprocess.run(cmd, check=False, cwd=REPO)


SWEEPS = {
    "shots": {f"shots_{k}": {"prior.n_shots": k} for k in (1, 2, 4, 8)},
    "resolution": {f"res_{r}": {"image.max_side": r} for r in (256, 384, 512, 768)},
    "terms": {
        "no_edge": {"scoring.w_edge": 0.0},
        "no_displacement": {"scoring.w_displacement": 0.0},
        "no_vlm_verdict": {"scoring.w_vlm": 0.0},
        "vlm_verdict_only": {
            "scoring.w_missing": 0.0, "scoring.w_extra": 0.0,
            "scoring.w_displacement": 0.0, "scoring.w_edge": 0.0, "scoring.w_vlm": 1.0,
        },
        "geometry_only": {"scoring.w_vlm": 0.0},
    },
    "model": {
        "gemini_flash": {"provider.name": "gemini", "provider.model": "gemini-2.0-flash",
                         "provider.api_key_env": "GEMINI_API_KEY"},
        "qwen25vl_72b": {"provider.name": "openrouter",
                         "provider.model": "qwen/qwen2.5-vl-72b-instruct:free",
                         "provider.api_key_env": "OPENROUTER_API_KEY"},
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sweep", choices=sorted(SWEEPS) + ["all"], default="all")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true", help="write configs, run nothing")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip inference and only recompute metrics from cached predictions")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / args.config).read_text())
    categories = args.categories or base.get("categories", ["pushpins"])
    names = sorted(SWEEPS) if args.sweep == "all" else [args.sweep]

    written = []
    for sweep in names:
        for variant, patch in SWEEPS[sweep].items():
            path = write_variant(base, f"{sweep}/{variant}".replace("/", "_"), patch)
            written.append((sweep, variant, path))
            print(f"[{sweep}] {variant} -> {path.relative_to(REPO)}")

    if args.dry_run:
        print(f"\n{len(written)} variant configs written; nothing executed.")
        return

    for sweep, variant, path in written:
        print(f"\n=== {sweep} / {variant} ===")
        run_variant(path, categories, skip_run=args.eval_only)

    # Collect everything into one table for the results section.
    rows = []
    for sweep, variant, _ in written:
        for cat in categories:
            m = REPO / "runs" / "sweep" / f"{sweep}_{variant}" / cat / "metrics.json"
            if m.exists():
                d = json.loads(m.read_text())
                rows.append({
                    "sweep": sweep, "variant": variant, "category": cat,
                    "auroc": d.get("auroc"), "auroc_logical": d.get("auroc_logical"),
                    "auroc_structural": d.get("auroc_structural"),
                    "f1_max": d.get("f1_max"),
                    "latency_p95": d.get("latency", {}).get("p95_s"),
                })
    out = REPO / "runs" / "sweep" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
