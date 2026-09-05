# TONIGHT — pushpins-only run for the ICCSDI 2026 deadline

Deadline: **5 September 2026, 11:59 PM IST**, via Microsoft CMT.

Work top to bottom. Each step names what to check before moving on. If a check
fails, the fix is listed — do not push past a failed check, because every later
number will be wrong in a way that looks plausible.

---

## Right now, in parallel

**A.** Start the pushpins download (mvtec.com → Research → Datasets → MVTec LOCO
AD → pushpins, ~1.0 GB). This is the long pole.

**B.** While it downloads, finish the environment. In PowerShell, in the project
folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gemini,dev]"
pytest -q
python -m keyprompt.cli selftest
```

**Check:** `19 passed`, then `all checks passed`. No dataset or key needed.

**C.** Get a Gemini key at aistudio.google.com. Create `.env` in the project root
(use PyCharm's Project panel, not Explorer — Windows hides file extensions):

```
GEMINI_API_KEY=your_key_here
```

---

## Step 1 — Place the data (2 min)

```
keyprompt-ad/data/mvtec_loco_anomaly_detection/pushpins/
    train/good/  validation/good/  test/good/
    test/logical_anomalies/  test/structural_anomalies/
    ground_truth/logical_anomalies/...
    defects_config.json
```

**Check:**

```powershell
python -c "import pathlib;p=pathlib.Path('data/mvtec_loco_anomaly_detection/pushpins/train/good');print(p.exists(), len(list(p.glob('*.png'))))"
```

Expect `True` and a few hundred. If `False`, you likely have a doubled folder
name — look at the actual path in PyCharm's Project panel.

---

## Step 2 — Annotations and prior (5 min)

```powershell
python -m keyprompt.cli --config configs/pushpins_only.yaml bootstrap --category pushpins --n-images 30
python -m keyprompt.cli --config configs/pushpins_only.yaml build-prior --category pushpins
```

**Check — this is the one that matters most.** Open a training image and count
the compartments in the tray. Both commands must report that same number:

```
surviving slots : 15 {'pushpin': 15}
slots:   15
```

**A wrong slot count fails silently.** Every downstream number will look
reasonable and mean nothing. Do not continue until it matches.

If it reports `0`, try in this order, adding `--overwrite` each time:

```powershell
... bootstrap --category pushpins --n-images 30 --invert --overwrite
... bootstrap --category pushpins --n-images 30 --invert --min-support 0.5 --overwrite
... bootstrap --category pushpins --n-images 30 --invert --min-area 5e-5 --overwrite
```

If it is non-zero but wrong, fall back to manual annotation
(`keyprompt annotate --category pushpins`, 4 images, a few minutes).

---

## Step 3 — Smoke test the API (3 min)

Do **not** launch the full run first.

```powershell
python -c "import yaml,pathlib; c=yaml.safe_load(open('configs/pushpins_only.yaml')); c['limit_per_split']=3; c['output_root']='runs/smoketest'; pathlib.Path('configs/smoketest.yaml').write_text(yaml.safe_dump(c,sort_keys=False))"

python -m keyprompt.cli --config configs/smoketest.yaml run --category pushpins
python -m keyprompt.cli --config configs/smoketest.yaml evaluate --category pushpins
```

**Check, in order:**

1. No `!` characters in the run output (those mark API errors).
2. `parse success` near `1.000`.
3. **Scores differ between images.** If every score is identical, the model
   returned nothing usable and the full run would burn your quota for nothing.

If scores are all identical, inspect one raw response:

```powershell
python -c "import json;print(json.dumps(json.loads(open('runs/smoketest/pushpins/predictions.jsonl').readline()),indent=2))"
```

Look at `breakdown` and `reasoning`. If `detected` is empty every time, try
`image.max_side: 768` in the config.

---

## Step 4 — Full run (20–30 min, unattended)

```powershell
python -m keyprompt.cli --config configs/pushpins_only.yaml run --category pushpins
python -m keyprompt.cli --config configs/pushpins_only.yaml evaluate --category pushpins
```

Pacing is 4 s between calls to stay inside the free-tier limit. It is resumable
and cached: if it stops, rerun the identical command.

**Go do the paper's author fields while this runs.**

---

## Step 5 — The ablation that matters (fast — reuses the cache)

```powershell
python scripts/sweep.py --config configs/pushpins_only.yaml --sweep terms --categories pushpins
```

This is **nearly free in wall-clock time**: the `terms` sweep only changes
scoring weights, not the prompt, so every model response comes from the cache.
No new API calls.

It produces the `vlm_verdict_only` control — the number your paper's central
claim rests on. If KeyPrompt-AD does not clearly beat it, say so in the
discussion and reframe; do not bury it.

Optional if time allows (**these do cost new API calls**, roughly 25 min each):

```powershell
python scripts/sweep.py --config configs/pushpins_only.yaml --sweep shots --categories pushpins
```

Skip `resolution` and `model` tonight.

---

## Step 6 — Generate the paper assets (5 min)

```powershell
python scripts/make_method_figures.py --out ..\paper\

python scripts/make_paper_assets.py --runs runs/pushpins_study --sweep runs/sweep/summary.json --dataset data/mvtec_loco_anomaly_detection --out ..\paper\
```

---

## Step 7 — Finish the manuscript (45–60 min)

The draft is already scoped to a single category: title, abstract, scope
paragraph, tables, discussion, limitations and conclusion all say pushpins only.

1. Paste the generated `table_*.tex` over the placeholder `tabular` blocks.
2. Swap `fig_placeholder.pdf` for the generated qualitative figure.
3. Fill the title page: names, affiliation, email.
4. Fill Funding, Code availability (repo URL + commit hash), Author
   contributions.
5. **Rewrite the Discussion to match what you actually measured.** It is written
   conditionally right now. Say what happened, including if the control was not
   beaten.
6. **Delete the `\needsdata` and `\nd` macro definitions** near the top of the
   preamble. The build then fails if any placeholder survives. This is the safety
   net — do not skip it.
7. Rebuild:

```
pdflatex KeyPromptAD_ICCSDI2026.tex
bibtex   KeyPromptAD_ICCSDI2026
pdflatex KeyPromptAD_ICCSDI2026.tex
pdflatex KeyPromptAD_ICCSDI2026.tex
```

8. Check the page limit on the CMT portal and trim Related Work or Discussion if
   over.

---

## Step 8 — Submit

Upload the PDF to Microsoft CMT before **11:59 PM IST**.

---

## If you run out of time

Submit nothing rather than a paper with placeholders in it. A visible
`[AWAITING EXPERIMENT]` box in a Springer-indexed submission is worse for you
than missing this deadline; the work is good and there will be another venue.

The honest fallback ordering, best to worst:

1. Full pipeline + `terms` ablation, single category — the target.
2. Full pipeline, no ablation, control reported from a manual weight change.
3. Nothing. Not a draft with empty tables.
