# RUNBOOK — running KeyPrompt-AD end to end

Follow these in order. Each step has a check: if the check fails, stop and fix it
before continuing. Most wasted time in this pipeline comes from pushing past a
failed step and debugging three stages later.

Total hands-on time is roughly 30 minutes. Most of the wall-clock time is the
dataset download and the API calls, both of which run unattended.

---

## Step 0 — Environment

Unzip `keyprompt-ad.zip` and open a terminal in the folder containing
`pyproject.toml`.

**Windows (PowerShell)** — note there is no `python3` command on Windows:

```powershell
python --version          # expect 3.10-3.12
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gemini,dev]"
```

**macOS / Linux:**

```bash
python3 --version         # expect 3.10-3.12
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gemini,dev]"
```

Your prompt should now start with `(.venv)`. If it doesn't, the environment is
not active and everything after this will install to the wrong place.

**Check:**

```
python -c "import numpy, scipy, PIL, yaml, matplotlib; print('deps ok')"
```

### If that failed on Windows

| Message | Cause | Fix |
|---|---|---|
| `'python3' is not recognized` | `python3` is Linux/macOS only | use `python` |
| `'python' is not recognized`, or the Microsoft Store opens | Python not on PATH | use the launcher: `py -3.12 -m venv .venv` |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again (applies to this window only) |

Once `(.venv)` is showing, plain `python` is correct for every later command on
all platforms.

---

## Step 1 — Verify the code before spending anything

This needs no dataset, no API key and no network.

```bash
pytest -q
keyprompt selftest
```

**Expected:**

```
17 passed in 5.12s
...
all checks passed
```

If this fails, nothing downstream is worth attempting. It means the install is
broken, not that the method doesn't work.

---

## Step 2 — Get an API key

Pick one. All three have a free tier sufficient for the full benchmark.

| Provider | Where | Config to use |
|---|---|---|
| Google AI Studio | aistudio.google.com | `configs/default.yaml` (default) |
| OpenRouter | openrouter.ai | `configs/openrouter.yaml` |
| Groq | console.groq.com | edit `provider.name: groq` |

```bash
cp .env.example .env
# open .env and paste the key, e.g. GEMINI_API_KEY=...
```

**Rotate the old key first.** The one that appeared in your original script has
been in a chat window and should be treated as compromised.

**Check:** `.env` must not be committed.

```bash
git check-ignore -v .env       # should print a .gitignore match
```

---

## Step 3 — Download MVTec LOCO AD

From https://www.mvtec.com/company/research/datasets/mvtec-loco (free for
research, registration required, ~5.7 GB for all five categories).

Extract so the tree looks exactly like this:

```
keyprompt-ad/
└── data/mvtec_loco_anomaly_detection/
    ├── pushpins/
    │   ├── train/good/            *.png
    │   ├── validation/good/       *.png
    │   ├── test/good/             *.png
    │   ├── test/logical_anomalies/      *.png
    │   ├── test/structural_anomalies/   *.png
    │   ├── ground_truth/logical_anomalies/<stem>/000.png ...
    │   └── defects_config.json
    ├── screw_bag/  splicing_connectors/  breakfast_box/  juice_bottle/
```

**Check:**

```bash
ls data/mvtec_loco_anomaly_detection/pushpins/train/good | head -3
```

Start with **pushpins** alone (~1.0 GB) if you want to get a first number today.

---

## Step 4 — Reference annotations

Two routes. Do the automatic one first; it is faster and gives better tolerance
estimates.

### 4a. Automatic (recommended)

```bash
keyprompt bootstrap --category pushpins --n-images 30
```

**Expected:** something like

```
proposing keypoints on 30 normal training images using the 'blobs' proposer

consensus over 30 normal images
  surviving slots : 15 {'pushpin': 15}
  dropped clusters: 12 (below support threshold)
  mean support    : 0.93 of images
```

The slot count must match the real component count. For pushpins that is the
number of compartments.

**If `surviving slots : 0`,** the proposer is not seeing the components. Try, in
this order:

```bash
keyprompt bootstrap --category pushpins --n-images 30 --invert --overwrite
keyprompt bootstrap --category pushpins --n-images 30 --invert --min-support 0.5 --overwrite
keyprompt bootstrap --category pushpins --n-images 30 --invert --min-area 5e-5 --overwrite
```

`--invert` matters when components are darker than their background.

**If the slot count is wrong but non-zero,** fall back to 4b, or fix the
generated JSON in `annotations/pushpins/` by hand.

### 4b. Manual

```bash
keyprompt annotate --category pushpins
```

A window opens per reference image. Click each component; `1`–`9` switch class,
`g` starts a new group, `u` undoes, `w` writes and advances, `q` skips.

Use groups for components that must hold a fixed arrangement — this is what
enables the spacing checks.

**Check either route:**

```bash
ls annotations/pushpins/          # one JSON per annotated image
```

---

## Step 5 — Fit the normality graph

```bash
keyprompt build-prior --category pushpins
```

**Expected:**

```
built normality graph from 30 shots -> runs/.../pushpins/prior.json
  classes: {'pushpin': 15}
  slots:   15
```

Confirm `slots` equals the true component count. Everything downstream depends
on this being right, and a wrong prior fails silently rather than loudly.

---

## Step 6 — Smoke test against the real API

Do not launch the full run first. Check that the model responds and parses.

```bash
# temporary 3-images-per-subset run
python - <<'EOF'
import yaml, pathlib
c = yaml.safe_load(open('configs/default.yaml'))
c['limit_per_split'] = 3
c['categories'] = ['pushpins']
c['output_root'] = 'runs/smoketest'
pathlib.Path('configs/smoketest.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
EOF

keyprompt --config configs/smoketest.yaml run      --category pushpins
keyprompt --config configs/smoketest.yaml evaluate --category pushpins
```

**Look for, in order:**

1. No `!` markers in the run output — those flag API errors.
2. `parse success` close to `1.000` in the evaluate output. Below ~0.9 means the
   model is not honouring the JSON schema; try a different model before
   proceeding.
3. Scores that are **not all identical**. Identical scores mean the model
   returned no usable detections, and the run is worthless.

Inspect one response directly:

```bash
head -1 runs/smoketest/pushpins/predictions.jsonl | python -m json.tool
```

The `breakdown` field should show non-zero terms and `defect_points` should be
populated for anomalous images.

---

## Step 7 — Full run

```bash
keyprompt run      --category pushpins
keyprompt evaluate --category pushpins
```

Budget roughly 10–20 minutes per category on a free tier: pacing is set to
4 s between calls to stay inside typical rate limits, so a few hundred test
images takes that long. It is resumable — if it stops, rerun the identical
command and it continues from where it stopped. Responses are cached, so
re-evaluating costs nothing.

Then the remaining categories:

```bash
for c in screw_bag splicing_connectors breakfast_box juice_bottle; do
  keyprompt bootstrap   --category $c --n-images 30
  keyprompt build-prior --category $c
  keyprompt run         --category $c
  keyprompt evaluate    --category $c
done

keyprompt report
```

> `screw_bag` and `juice_bottle` have several component classes. The automatic
> proposer is single-class, so bootstrap then fix the class labels in
> `annotations/<category>/*.json`, or annotate those two by hand.

---

## Step 8 — Ablations

```bash
python scripts/sweep.py --sweep terms       --categories pushpins
python scripts/sweep.py --sweep shots       --categories pushpins
python scripts/sweep.py --sweep resolution  --categories pushpins
python scripts/sweep.py --sweep model       --categories pushpins
```

Run `terms` first. It contains the `vlm_verdict_only` control, which measures
what the model achieves with the geometric audit switched off. If the full
system does not clearly beat that control, the contribution is the prompt rather
than the method, and the paper's framing has to change. Better to learn this now
than after a reviewer does.

---

## Step 9 — Paper assets

```bash
python scripts/make_method_figures.py --out ../paper/

python scripts/make_paper_assets.py \
    --runs runs/gemini-flash-4shot \
    --sweep runs/sweep/summary.json \
    --dataset data/mvtec_loco_anomaly_detection \
    --out ../paper/
```

Paste the generated `table_*.tex` over the placeholder blocks in the manuscript,
then **delete the `\needsdata` and `\nd` macro definitions** near the top of the
preamble. The build will then fail if any placeholder survives, which is the
safety net.

---

## Step 10 — Push

```bash
bash scripts/git_init.sh
```

It stages everything and prints the file list for review. Confirm `.env`,
`data/`, `runs/` and `.cache/` are absent, then commit and push.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no prior found` | skipped step 5 | run `build-prior` |
| `none of the N annotations match images` | annotations from a different category or dataset root | re-run `bootstrap` |
| `surviving slots : 0` | proposer sees no components | `--invert`, lower `--min-support`, lower `--min-area` |
| All scores identical | model returned no detections | check `parse success`; try another model |
| Repeated `429` | free-tier rate limit | raise `provider.min_seconds_between_calls` |
| Run seems stuck | pacing delay between calls | expected; it is resumable, leave it |
| AUROC near 0.5 | see below | inspect the prompt and a raw response |

**AUROC near 0.5** is the failure worth diagnosing carefully. Read
`runs/<category>/prompt.txt` — the layout specification should list the correct
number of slots at plausible coordinates. Then read a few
`predictions.jsonl` rows. The three common causes are: a wrong prior (slot count
does not match reality), a coordinate-convention mismatch (the model answering in
pixels or on a 0–1000 grid, though the parser handles the common cases), or the
model simply not grounding well at that resolution — try `image.max_side: 768`.
