# KeyPrompt-AD

Few-shot logical anomaly detection for industrial parts, using an off-the-shelf
vision-language model and four annotated reference images per category. No
training, no fine-tuning, no labelled defects.

Benchmarked on [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco).

---

## The problem this addresses

Supervised defect detectors need examples of defects. On a real line you rarely
have them: a nut that failed to weld, a hole that was never punched, or a
component sitting two millimetres out of position each appear a handful of times
a year, and by the time you have collected a training set the product has
changed. What you always have is the opposite — plenty of correct parts.

The harder half of the problem is that "defective" is not a fixed appearance. A
part can be defective because something is *absent*, or because two components
that should be a fixed distance apart are not. Nothing about the pixels in any
one region looks wrong; the fault lives in the relationship between regions.
Texture-based anomaly detectors, which look for locally unusual patches, are
structurally poor at this. MVTec LOCO AD calls these *logical* anomalies and
separates them from *structural* ones (scratches, dents) precisely because the
gap in performance between the two is so large.

## Approach

Annotate a small number of correct parts with keypoints — one point per
component, optionally tagged with a group — and let the rest follow from that.

**1. Normality graph.** The reference annotations are aligned into a common
frame and condensed into a compact statistical model: how many instances of each
component class to expect, where each one belongs, how much positional spread is
normal, and what the characteristic spacing is between every pair of slots. That
last part is what catches the "everything present but spaced wrongly" defect.

**2. Prompt grounding.** The graph is rendered into an explicit layout
specification and placed in the prompt alongside the reference images and their
exact coordinates. The model is not asked *"does this look anomalous?"* — a
question it answers poorly and unstably. It is asked to report where each
component is, against a specification it has been handed. The spec is generated,
not hand-written, so pointing the method at a new part means re-annotating, not
re-prompting.

**3. Geometric audit.** The returned coordinates are matched one-to-one against
the prior slots (optimal assignment, gated by a radius), after a global pose
correction so that a shifted fixture is not mistaken for a fault. Unmatched
slots are missing components, unmatched detections are extras, matched pairs
contribute a displacement in units of the learned tolerance, and observed
spacings are checked against the learned ones.

**4. Continuous score.** Those four terms, plus the model's own verdict at low
weight, fuse into a scalar in `[0, 1]`. This matters more than it sounds: a VLM
emits a categorical verdict, which gives you one operating point and no ROC
curve, and therefore no honest comparison against the unsupervised baselines
everyone else reports. The geometric layer supplies the missing continuum.

The design keeps the language model doing what it is good at — grounding named
objects in an image — and moves the metric decision, which it is bad at, into
deterministic code you can inspect and unit-test.

<img width="1748" height="494" alt="image" src="https://github.com/user-attachments/assets/09fe09d0-9064-471e-a7be-be1f0cf5425e" />


## Install

```bash
git clone https://github.com/<you>/keyprompt-ad.git
cd keyprompt-ad
python3 -m venv .venv && source .venv/bin/activate   # Windows: python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[gemini,dev]"

cp .env.example .env      # then paste your key into .env
```

`.env` is gitignored. Do not commit keys — and if a key has ever appeared in a
notebook, an issue, or a chat window, rotate it before using it here.

Free API options, all usable for a full benchmark run:

| Provider | Config | Notes |
|---|---|---|
| Google AI Studio | `provider.name: gemini` | Strongest spatial grounding, native JSON mode. Default. |
| OpenRouter | `provider.name: openrouter` | Free open-weight VLMs (Qwen2.5-VL, Llama vision). Use for the cross-model ablation. |
| Groq | `provider.name: groq` | Lowest latency; good for the throughput argument. |
| — | `provider.name: echo` | Offline stub, no key needed. For smoke tests and CI. |

## Data

Download MVTec LOCO AD from MVTec (free for research; CC BY-NC-SA 4.0) and
extract so the tree looks like:

```
data/mvtec_loco_anomaly_detection/
  pushpins/
    train/good/            validation/good/
    test/good/             test/logical_anomalies/     test/structural_anomalies/
    ground_truth/logical_anomalies/<stem>/000.png ...
    defects_config.json
  screw_bag/  splicing_connectors/  breakfast_box/  juice_bottle/
```

Start with `pushpins` and `screw_bag`. Both are close analogues of the
missing-nut and component-count problems this method was designed for.

## Usage

```bash
# 0. verify everything works. No dataset, no API key, no network.
keyprompt selftest

# 1a. derive annotations automatically from 30 normal training images (preferred)
keyprompt bootstrap   --category pushpins --n-images 30

# 1b. or annotate by hand (click points; 1..9 = class, g = new group,
#     u = undo, w = write and advance). A few minutes per category.
keyprompt annotate    --category pushpins

# 2. fit the normality graph
keyprompt build-prior --category pushpins

# 3. run inference over the full test split (resumable; responses are cached)
keyprompt run         --category pushpins

# 4. metrics
keyprompt evaluate    --category pushpins

# 5. table across all categories
keyprompt report
```

Responses are content-addressed and cached, so a rerun after a rate-limit stall
costs nothing and re-evaluation is instant. Predictions stream to disk line by
line, so an interrupted run resumes where it stopped — just rerun the same
command. Add `--overwrite` to start clean.

Two paths worth knowing about:

- **Annotations live in `annotations/`, not `runs/`.** They are input data, so
  an ablation sweep that varies `output_root` reuses one annotation set instead
  of demanding a fresh one per variant.
- **`prior.n_shots` controls the prompt, not the prior.** The normality graph is
  fitted from every annotation available; only `n_shots` images are sent in the
  prompt. Bootstrapping from 30 images therefore gives sharper tolerances
  without inflating per-call cost.

## Getting the reference annotations

Hand-clicking is supported but is the weakest link in the design — not because
it is slow, but because four hand-placed points are a poor sample from which to
estimate a positional tolerance. `bootstrap` does better on both counts.

It proposes keypoints on many normal training images, aligns them into a common
frame, clusters the results, and keeps only the clusters that recur in at least
`--min-support` of images. Real component slots survive; detector artefacts do
not. Tolerances then come from tens of observations rather than four.

```bash
keyprompt bootstrap --category pushpins --n-images 30              # classical proposer
keyprompt bootstrap --category pushpins --proposer vlm             # model-proposed
keyprompt bootstrap --category pushpins --invert --min-support 0.5 # if nothing survives
```

Two proposers:

- **`blobs`** (default) — connected components on an Otsu-thresholded image. No
  model, no network, deterministic. Good when components contrast with the
  background; use `--invert` when they are darker than it.
- **`vlm`** — the model locates components from the text normality statement
  alone, with no coordinates supplied. Fully hands-off, but the prior is then
  built from the model's own detections and later used to grade that same
  model, so errors are absorbed rather than exposed. **Report this variant
  separately from the manually-annotated one; do not pool the two.**

Neither proposer knows which cluster belongs to which class when a category has
several. For `screw_bag` and `juice_bottle`, bootstrap first and then fix the
class labels by hand — still far less work than placing every point.

The prior is built only from `train/good`. Nothing in the pipeline reads
`ground_truth/`, which exists solely for anomalous test images; using it to
build the normality graph would be test-set leakage.

## What gets measured

**Detection.** Image-level AUROC and average precision, best achievable F1, and
accuracy at that operating point. AUROC is reported separately for the logical
and structural subsets, since the method targets the logical half and hiding
that behind a single average would be misleading.

**Localisation.** The official LOCO metric (saturated per-region overlap)
assumes dense masks; this system emits points, so a point-based analogue is used
and labelled as such. *Region recall* is the fraction of annotated defect
regions containing at least one predicted point; *point precision* is the
fraction of predicted points falling inside some annotated region. Both are
computed over anomalous images only, with a small dilation tolerance because
missing-component regions mark empty space.

**Efficiency.** Latency mean/p50/p95 and token counts, measured on uncached
calls only, plus the parse success rate — how often the model returned JSON that
survived parsing. That number belongs in the paper; it is the honest measure of
how brittle prompt-based inspection is in practice.

## Ablations

```bash
python scripts/sweep.py --sweep shots       # 1 / 2 / 4 / 8 reference images
python scripts/sweep.py --sweep resolution  # 256 / 384 / 512 / 768 px
python scripts/sweep.py --sweep terms       # which scoring term does the work
python scripts/sweep.py --sweep model       # Gemini vs an open-weight VLM
```

The `terms` sweep includes a `vlm_verdict_only` variant, which is the honest
control: it measures what the model achieves when the geometric audit is
switched off entirely. If the full system does not clearly beat it, the
contribution is the prompt, not the method, and the paper should say so.

## Baselines to compare against

Reviewers will ask. Run these on the same split via
[anomalib](https://github.com/openvinotoolkit/anomalib):

- **PatchCore** — the standard memory-bank baseline; strong on structural
  anomalies, weak on logical ones, which is the contrast this work is built on.
- **PaDiM** — cheaper, similar profile.
- **WinCLIP / AnomalyCLIP** — the CLIP-based few-shot line of work; the nearest
  neighbour to this method in spirit.

Published LOCO numbers exist for all of these, but reproducing them on your own
split is worth the afternoon: it removes any question about protocol drift.

## Repository layout

```
keyprompt-ad/
├── README.md
├── pyproject.toml            installable; provides the `keyprompt` command
├── requirements.txt
├── .env.example              copy to .env and add your key
├── .gitignore                excludes .env, data/, runs/, .cache/
├── LICENSE
│
├── configs/
│   ├── default.yaml          the main benchmark run
│   ├── smoke.yaml            offline stub provider, 3 images per subset
│   ├── openrouter.yaml       cross-model check on an open-weight VLM
│   └── categories/           per-category normality statements (plain English)
│       ├── pushpins.yaml     screw_bag.yaml       breakfast_box.yaml
│       └── juice_bottle.yaml splicing_connectors.yaml
│
├── scripts/
│   ├── sweep.py              ablations: shots, resolution, terms, model
│   └── git_init.sh           first push, with a staging safety check
│
├── src/keyprompt/
│   ├── cli.py                selftest, bootstrap, annotate, build-prior,
│   │                         run, evaluate, report
│   ├── config.py             run configuration, serialised with every result
│   ├── data/loco.py          LOCO reader, splits, per-region ground truth
│   ├── annotate/
│   │   ├── schema.py         keypoint annotation format
│   │   ├── tool.py           click-to-annotate tool
│   │   └── auto.py           automatic proposal + consensus filtering
│   ├── prior/
│   │   ├── geometry.py       similarity fit, gated optimal assignment
│   │   └── graph.py          the normality graph
│   ├── prompting/
│   │   ├── builder.py        prompt assembly from graph + category spec
│   │   └── schema.py         response schema and a forgiving parser
│   ├── pipeline/
│   │   ├── detector.py       one image in, score and coordinates out
│   │   └── scoring.py        geometric audit and score fusion
│   ├── providers/
│   │   ├── base.py           caching, retries, request pacing
│   │   └── backends.py       Gemini, OpenRouter, Groq, offline stub
│   ├── eval/metrics.py       detection, localisation, efficiency
│   └── viz.py                overlays and contact sheets
│
├── tests/                    17 tests; no key or dataset required
│
└── (generated, all gitignored)
    ├── data/                 MVTec LOCO, downloaded separately
    ├── annotations/<cat>/    reference keypoints — input data, kept stable
    ├── runs/<cat>/           prior.json, predictions.jsonl, metrics.json
    └── .cache/vlm/           content-addressed response cache
```

## Tests

```bash
keyprompt selftest    # end-to-end pipeline check, prints a pass/fail table
pytest -q             # 17 unit tests
```

No API key or dataset required for either. The tests cover the parts where a silent bug
would corrupt every reported number: pose recovery, assignment gating, the
prior's slot count, that a globally shifted part scores low, and that AUROC
handles ties correctly.

## Limitations

Worth stating plainly, and worth stating in the paper too.

- Per-image cost is an API call. This is not a 200 ms inline inspection system;
  it suits low-volume, high-mix lines and new-product introduction, where the
  alternative is waiting months to collect a training set.
- Accuracy depends on the backbone's spatial grounding, which varies a lot
  between models. The `model` sweep exists to quantify that dependence rather
  than paper over it.
- Categories whose components move freely (`screw_bag`) get little from the
  positional prior; there, counting carries the signal and the geometry terms
  contribute less. That contrast is itself a result.
- Reference annotations can be bootstrapped, but the automatic proposers are
  single-class by default; multi-class categories still need a short human pass
  to assign labels. The method is few-shot, not zero-shot.
- Results in this repository have been validated on synthetic point sets only.
  The geometry, scoring, consensus and metric code is unit-tested; no numbers on
  real MVTec LOCO images have been produced yet.

## Licence

MIT for the code. MVTec LOCO AD is licensed separately by MVTec Software GmbH
(CC BY-NC-SA 4.0); no dataset images are included in this repository.
