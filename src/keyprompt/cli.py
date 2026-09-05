"""Command line entry point.

    keyprompt selftest                          # no dataset or key needed
    keyprompt bootstrap   --category pushpins   # automatic, or:
    keyprompt annotate    --category pushpins   # manual
    keyprompt build-prior --category pushpins
    keyprompt run         --category pushpins
    keyprompt evaluate    --category pushpins
    keyprompt report

Each stage writes its artefacts to disk, so a failed or rate-limited run can be
resumed rather than restarted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .annotate.auto import consensus_annotations, propose_blobs, propose_vlm
from .annotate.schema import ImageAnnotation, load_annotation_set
from .config import RunConfig, load_category
from .data.loco import CATEGORIES, LocoCategory
from .dotenv import load_dotenv
from .eval.metrics import LocalisationCase, localisation_scores, summarise
from .pipeline.detector import KeyPromptDetector, Prediction
from .prior.graph import NormalityGraph
from .prompting.builder import build_context_prompt
from .providers.backends import build_provider


def _paths(cfg: RunConfig, category: str) -> dict:
    root = Path(cfg.output_root) / category
    return {
        "root": root,
        "annotations": Path(cfg.annotations_root) / category,
        "prior": root / "prior.json",
        "predictions": root / "predictions.jsonl",
        "metrics": root / "metrics.json",
        "prompt": root / "prompt.txt",
        "figures": root / "figures",
    }


def _category_cfg(category: str):
    p = Path("configs/categories") / f"{category}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"missing category spec: {p}")
    return load_category(p)


# -- commands ------------------------------------------------------------


def cmd_annotate(args) -> None:
    from .annotate.tool import KeypointAnnotator

    cfg = RunConfig.from_yaml(args.config)
    cat_cfg = _category_cfg(args.category)
    paths = _paths(cfg, args.category)

    cat = LocoCategory.open(cfg.dataset_root, args.category)
    shots = cat.reference_shots(cfg.prior.n_shots, seed=cfg.seed)

    print(f"Annotating {len(shots)} reference images for '{args.category}'.")
    print(f"Classes: {cat_cfg.component_classes}")
    tool = KeypointAnnotator(cat_cfg.component_classes, paths["annotations"])
    for s in shots:
        if (paths["annotations"] / f"{s.stem}.json").exists() and not args.overwrite:
            print(f"  skipping {s.stem} (already annotated)")
            continue
        tool.annotate(s.image_path, s.uid, args.category, use_groups=args.groups)


def cmd_bootstrap(args) -> None:
    """Derive reference annotations automatically from normal training images.

    Cheaper than hand-clicking, and better: tolerances estimated from tens of
    images are far more trustworthy than ones estimated from four.
    """
    cfg = RunConfig.from_yaml(args.config)
    cat_cfg = _category_cfg(args.category)
    paths = _paths(cfg, args.category)

    cat = LocoCategory.open(cfg.dataset_root, args.category)
    pool = cat.train_normal()[: args.n_images]
    if not pool:
        sys.exit(f"no training images found for {args.category}")
    print(f"proposing keypoints on {len(pool)} normal training images "
          f"using the '{args.proposer}' proposer")

    images = [s.load_image() for s in pool]

    if args.proposer == "blobs":
        cls = (cat_cfg.component_classes or ["component"])[0]
        proposer = lambda im: propose_blobs(  # noqa: E731
            im, cls=cls, invert=args.invert,
            min_area_frac=args.min_area, max_area_frac=args.max_area,
        )
    else:
        provider = build_provider(
            cfg.provider,
            cache_dir=Path(cfg.cache_root) / cfg.provider.name / "bootstrap",
            use_cache=cfg.use_cache,
        )
        proposer = lambda im: propose_vlm(im, provider, cat_cfg, cfg.image)  # noqa: E731

    anns, report = consensus_annotations(
        images, proposer,
        min_support=args.min_support,
        cluster_radius=args.cluster_radius,
    )
    print("\n" + report.describe())

    if report.n_slots == 0:
        sys.exit(
            "\nno slots survived the consensus filter. Try a lower --min-support, "
            "a larger --cluster-radius, or --invert if components are darker "
            "than the background."
        )

    out = paths["annotations"]
    if out.exists() and any(out.glob("*.json")) and not args.overwrite:
        sys.exit(f"\n{out} already contains annotations; pass --overwrite to replace them")
    for i, (a, s_) in enumerate(zip(anns, pool)):
        a.image_uid = s_.uid
        a.category = args.category
        a.to_json(out / f"{s_.stem}.json")
    print(f"\nwrote {len(anns)} annotations to {out}")
    print("Review a few before building the prior. Automatic proposals are a "
          "starting point, not ground truth.")


def cmd_selftest(args) -> None:
    """Exercise the whole pipeline on synthetic data.

    No dataset, no API key, no network. This is the first thing to run after
    cloning: it confirms the geometry, prior, prompting, provider plumbing,
    scoring and metrics all fit together, so that a later failure on real data
    can be attributed to the data or the model rather than the code.
    """
    import numpy as np
    from PIL import Image

    from .annotate.auto import consensus_annotations
    from .annotate.schema import Keypoint
    from .config import CategoryConfig, ImageConfig, RunConfig as RC
    from .eval.metrics import roc_auc, summarise
    from .pipeline.scoring import score_detection
    from .providers.backends import build_provider as _bp

    cfg = RC()
    cfg.provider.name = "echo"
    cfg.provider.min_seconds_between_calls = 0.0
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and condition
        print(f"  [{'ok ' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    grid = [(0.12 + 0.19 * c, 0.20 + 0.30 * r) for r in range(3) for c in range(5)]
    rng = np.random.default_rng(0)

    print("\n1. normality graph from manual-style annotations")
    anns = [
        ImageAnnotation(
            f"ref{k}", "synthetic", 1000, 600,
            [Keypoint("pushpin", x + rng.normal(0, 0.004), y + rng.normal(0, 0.004),
                      group=f"g{i // 5}") for i, (x, y) in enumerate(grid)],
        )
        for k in range(4)
    ]
    graph = NormalityGraph.build(anns, "synthetic")
    check("slot count matches layout", graph.total_slots() == len(grid),
          f"{graph.total_slots()}/{len(grid)}")

    print("\n2. automatic bootstrap from a deliberately noisy proposer")
    frames = []
    for _ in range(30):
        pts = [("pushpin", x + rng.normal(0, 0.006), y + rng.normal(0, 0.006))
               for (x, y) in grid if rng.random() >= 0.25]
        pts += [("pushpin", rng.uniform(0, 1), rng.uniform(0, 1))
                for _ in range(int(rng.integers(0, 4)))]
        frames.append(pts)
    it = iter(frames)
    auto_anns, report = consensus_annotations(
        [Image.new("RGB", (1000, 600)) for _ in frames], lambda im: next(it)
    )
    auto_graph = NormalityGraph.build(auto_anns, "synthetic")
    check("consensus recovers every slot despite 25% miss rate",
          auto_graph.total_slots() == len(grid), f"{auto_graph.total_slots()}/{len(grid)}")
    check("spurious detections rejected", report.dropped_clusters > 0,
          f"{report.dropped_clusters} dropped")

    print("\n3. prompt assembly")
    cat_cfg = CategoryConfig(name="synthetic", component_classes=["pushpin"],
                             normality_statement="Fifteen pushpins in a fixed grid.")
    prompt = build_context_prompt(cat_cfg, graph)
    check("layout specification generated", "expected count = 15" in prompt,
          f"{len(prompt)} chars")
    check("edge list stays bounded", len(prompt) < 20000)

    print("\n4. geometric scoring")
    sc = cfg.scoring
    normal = score_detection({"pushpin": grid}, graph, sc, "OK", 0.0)
    missing = score_detection({"pushpin": [p for i, p in enumerate(grid) if i != 7]},
                              graph, sc, "NOT OK", 0.9)
    shifted = score_detection({"pushpin": [(x + (0.09 if i == 3 else 0), y)
                                           for i, (x, y) in enumerate(grid)]},
                              graph, sc, "NOT OK", 0.8)
    extra = score_detection({"pushpin": list(grid) + [(0.55, 0.85)]}, graph, sc, "NOT OK", 0.7)
    moved = score_detection({"pushpin": [(x + 0.03, y - 0.02) for x, y in grid]},
                            graph, sc, "OK", 0.0)
    check("missing component detected", missing.n_missing == 1 and missing.score > normal.score,
          f"{normal.score:.3f} -> {missing.score:.3f}")
    check("displaced component detected", shifted.score > normal.score,
          f"{normal.score:.3f} -> {shifted.score:.3f}")
    check("extra component detected", extra.n_extra == 1 and extra.score > normal.score,
          f"{normal.score:.3f} -> {extra.score:.3f}")
    check("global shift is not a defect", moved.n_missing == 0 and moved.score < 0.10,
          f"score {moved.score:.3f}")

    print("\n5. provider plumbing (offline stub)")
    provider = _bp(cfg.provider, cache_dir=None, use_cache=False)
    resp = provider.generate(["hello", Image.new("RGB", (64, 64))])
    check("provider returns a parseable response", not resp.error and "verdict" in resp.text)

    print("\n6. metrics")
    labels = [0, 1, 1, 1]
    scores = [normal.score, missing.score, shifted.score, extra.score]
    auc = roc_auc(labels, scores)
    summary = summarise(labels, scores,
                        ["good", "logical_anomalies", "logical_anomalies", "logical_anomalies"],
                        [1.0, 1.1, 1.2, 1.3])
    check("AUROC separates the defect modes", auc == 1.0, f"AUROC {auc:.3f}")
    check("summary reports a logical subset", "auroc_logical" in summary)

    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)


def cmd_build_prior(args) -> None:
    cfg = RunConfig.from_yaml(args.config)
    paths = _paths(cfg, args.category)
    anns = load_annotation_set(paths["annotations"])
    if not anns:
        sys.exit(f"no annotations in {paths['annotations']}; run 'annotate' first")

    graph = NormalityGraph.build(
        anns, args.category, align=cfg.prior.align, min_sigma=cfg.prior.min_sigma
    )
    graph.save(paths["prior"])
    print(f"built normality graph from {len(anns)} shots -> {paths['prior']}")
    print(f"  classes: {graph.expected_counts()}")
    print(f"  slots:   {graph.total_slots()}")


def cmd_run(args) -> None:
    cfg = RunConfig.from_yaml(args.config)
    cat_cfg = _category_cfg(args.category)
    paths = _paths(cfg, args.category)
    paths["root"].mkdir(parents=True, exist_ok=True)

    if not paths["prior"].exists():
        sys.exit("no prior found; run 'build-prior' first")
    graph = NormalityGraph.load(paths["prior"])
    anns: List[ImageAnnotation] = load_annotation_set(paths["annotations"])

    cat = LocoCategory.open(cfg.dataset_root, args.category)
    by_stem = {s.stem: s for s in cat.train_normal()}

    # The prior is fitted from every available annotation (bootstrap may have
    # produced tens of them). The prompt, by contrast, carries only n_shots
    # images: sending all of them would blow up context and cost, and would
    # silently contradict the shot count the run is supposed to be testing.
    usable = [a for a in anns if a.image_uid.split("/")[-1] in by_stem]
    if not usable:
        sys.exit(
            f"none of the {len(anns)} annotations match images under "
            f"{cfg.dataset_root}/{args.category}/train/good"
        )
    # Prefer the most complete annotations, then break ties deterministically so
    # a rerun sends the same shots.
    usable.sort(key=lambda a: (-len(a.keypoints), a.image_uid))
    shots = usable[: cfg.prior.n_shots]
    if len(usable) < cfg.prior.n_shots:
        print(
            f"warning: n_shots={cfg.prior.n_shots} but only {len(usable)} "
            f"annotations are available; using {len(usable)}"
        )
    print(f"prior fitted from {len(anns)} annotations; prompting with {len(shots)} shots")

    ref_images = [by_stem[a.image_uid.split("/")[-1]].load_image() for a in shots]
    ref_anns = list(shots)

    provider = build_provider(
        cfg.provider,
        cache_dir=Path(cfg.cache_root) / cfg.provider.name / cfg.provider.model.replace("/", "_"),
        use_cache=cfg.use_cache,
    )
    detector = KeyPromptDetector(cfg, cat_cfg, graph, ref_images, ref_anns, provider)
    paths["prompt"].write_text(detector._context + "\n\n" + detector._query)

    samples = cat.test(limit_per_subset=cfg.limit_per_split)
    done = set()
    if paths["predictions"].exists() and not args.overwrite:
        for line in paths["predictions"].read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])
        print(f"resuming: {len(done)} predictions already on disk")

    mode = "w" if args.overwrite else "a"
    with paths["predictions"].open(mode) as fh:
        for i, s in enumerate(samples, start=1):
            if s.uid in done:
                continue
            pred: Prediction = detector.predict(s)
            fh.write(json.dumps(pred.to_dict()) + "\n")
            fh.flush()
            flag = "!" if pred.error else " "
            print(
                f"[{i}/{len(samples)}]{flag} {s.subset:22s} {s.stem:12s} "
                f"score={pred.score:.3f} verdict={pred.verdict:7s} "
                f"{pred.latency_s:.1f}s{' (cached)' if pred.cached else ''}"
            )
    print(f"wrote {paths['predictions']}")


def cmd_evaluate(args) -> None:
    cfg = RunConfig.from_yaml(args.config)
    paths = _paths(cfg, args.category)
    if not paths["predictions"].exists():
        sys.exit("no predictions found; run 'run' first")

    rows = [json.loads(l) for l in paths["predictions"].read_text().splitlines() if l.strip()]
    summary = summarise(
        labels=[r["label"] for r in rows],
        scores=[r["score"] for r in rows],
        subsets=[r["subset"] for r in rows],
        latencies=[r["latency_s"] for r in rows if not r.get("cached")],
        prompt_tokens=[r.get("prompt_tokens") for r in rows],
        output_tokens=[r.get("output_tokens") for r in rows],
        parse_ok=[r.get("parse_ok", True) for r in rows],
    )

    cat = LocoCategory.open(cfg.dataset_root, args.category)
    by_uid = {s.uid: s for s in cat.test(limit_per_subset=cfg.limit_per_split)}
    cases = []
    for r in rows:
        if r["label"] != 1:
            continue
        s = by_uid.get(r["uid"])
        if s is None:
            continue
        regions = s.load_gt_regions()
        if regions:
            cases.append(
                LocalisationCase(points=[tuple(p) for p in r.get("defect_points", [])],
                                 regions=regions)
            )
    if cases:
        summary["localisation"] = localisation_scores(cases, tolerance_px=args.tolerance)

    summary["config"] = cfg.to_dict()
    summary["category"] = args.category
    paths["metrics"].write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== {args.category} ===")
    print(f"  images            {summary['n']} ({summary['n_normal']} ok / {summary['n_anomalous']} nok)")
    print(f"  AUROC             {summary['auroc']:.4f}")
    print(f"  AUROC logical     {summary.get('auroc_logical', float('nan')):.4f}")
    print(f"  AUROC structural  {summary.get('auroc_structural', float('nan')):.4f}")
    print(f"  avg precision     {summary['average_precision']:.4f}")
    print(f"  F1-max            {summary['f1_max']:.4f}")
    op = summary["operating_point"]
    print(f"  at best F1        acc={op['accuracy']:.3f} prec={op['precision']:.3f} rec={op['recall']:.3f}")
    if "localisation" in summary:
        loc = summary["localisation"]
        print(f"  region recall     {loc['region_recall']:.4f}  over {loc['n_regions']} regions")
        print(f"  point precision   {loc['point_precision']:.4f}")
    if "latency" in summary:
        lat = summary["latency"]
        print(f"  latency           mean {lat['mean_s']:.2f}s  p95 {lat['p95_s']:.2f}s")
    print(f"  parse success     {summary.get('parse_success_rate', 1.0):.3f}")
    print(f"\nwrote {paths['metrics']}")


def cmd_report(args) -> None:
    cfg = RunConfig.from_yaml(args.config)
    rows = []
    for c in cfg.categories:
        p = _paths(cfg, c)["metrics"]
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        sys.exit("no metrics found")

    def fmt(value, width: int = 5, suffix: str = "") -> str:
        """Render a metric that may be absent, null or NaN without breaking the table."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return "-".rjust(width + len(suffix))
        if f != f:  # NaN
            return "-".rjust(width + len(suffix))
        return f"{f:{width}.2f}{suffix}" if suffix else f"{f:{width}.3f}"

    hdr = (f"| {'category':22s} | AUROC | logic | struct |   AP  | F1max | lat p95 |")
    rule = "|" + "-" * (len(hdr) - 2) + "|"
    print(hdr)
    print(rule)
    for r in rows:
        print(
            f"| {r.get('category', '?'):22s} "
            f"| {fmt(r.get('auroc'))} "
            f"| {fmt(r.get('auroc_logical'))} "
            f"| {fmt(r.get('auroc_structural'))} "
            f"| {fmt(r.get('average_precision'))} "
            f"| {fmt(r.get('f1_max'))} "
            f"| {fmt((r.get('latency') or {}).get('p95_s'), width=6, suffix='s')} |"
        )

    valid = [r["auroc"] for r in rows
             if isinstance(r.get("auroc"), (int, float)) and r["auroc"] == r["auroc"]]
    if valid:
        print(f"\nmean AUROC over {len(valid)} categories: {sum(valid) / len(valid):.4f}")
    else:
        print("\nno usable AUROC values found")


# -- parser --------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="keyprompt", description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("annotate", help="click keypoints on the reference shots")
    a.add_argument("--category", required=True, choices=CATEGORIES)
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--groups", action="store_true", default=True)
    a.set_defaults(func=cmd_annotate)

    st = sub.add_parser("selftest", help="verify the pipeline with no dataset or API key")
    st.set_defaults(func=cmd_selftest)

    s_ = sub.add_parser("bootstrap", help="derive annotations automatically (no clicking)")
    s_.add_argument("--category", required=True, choices=CATEGORIES)
    s_.add_argument("--proposer", choices=["blobs", "vlm"], default="blobs")
    s_.add_argument("--n-images", type=int, default=30,
                    help="normal training images to pool; more is better")
    s_.add_argument("--min-support", type=float, default=0.6,
                    help="fraction of images a cluster must appear in to survive")
    s_.add_argument("--cluster-radius", type=float, default=0.05)
    s_.add_argument("--invert", action="store_true",
                    help="components are darker than the background")
    s_.add_argument("--min-area", type=float, default=2e-4)
    s_.add_argument("--max-area", type=float, default=5e-2)
    s_.add_argument("--overwrite", action="store_true")
    s_.set_defaults(func=cmd_bootstrap)

    b = sub.add_parser("build-prior", help="fit the normality graph")
    b.add_argument("--category", required=True, choices=CATEGORIES)
    b.set_defaults(func=cmd_build_prior)

    r = sub.add_parser("run", help="run inference over the test split")
    r.add_argument("--category", required=True, choices=CATEGORIES)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="compute metrics from saved predictions")
    e.add_argument("--category", required=True, choices=CATEGORIES)
    e.add_argument("--tolerance", type=int, default=12)
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("report", help="aggregate table across categories")
    t.set_defaults(func=cmd_report)

    args = p.parse_args(argv)

    # Make `.env` work the way the documentation promises. Real environment
    # variables take precedence, so an IDE run configuration or an exported
    # shell value is never overridden by a stale file.
    loaded = load_dotenv()
    if loaded and args.command not in ("selftest",):
        print(f"loaded environment from {loaded}")

    try:
        args.func(args)
    except KeyboardInterrupt:
        # Predictions stream to disk line by line, so an interrupted run resumes.
        print("\ninterrupted; rerun the same command to resume where it stopped")
        sys.exit(130)
    except FileNotFoundError as exc:
        sys.exit(f"\n{exc}")
    except BrokenPipeError:
        # Raised when output is piped into a command that exits early, such as
        # `keyprompt run | head`. Not an error worth a traceback.
        try:
            sys.stdout.close()
        finally:
            sys.exit(0)


if __name__ == "__main__":
    main()
