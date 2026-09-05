# PyCharm setup guide

A complete walkthrough for running this project in PyCharm, assuming nothing is
installed yet. Menu paths are for recent PyCharm versions on Windows and Linux;
on macOS, **File → Settings** is **PyCharm → Settings** (or **Preferences**).

---

## Part 1 — What to download

Four things. Get all four before starting; two of them take a while.

| # | What | Where | Size | Notes |
|---|---|---|---|---|
| 1 | **Python 3.10–3.12** | python.org/downloads | ~30 MB | On Windows, tick **"Add Python to PATH"** during install |
| 2 | **PyCharm** | jetbrains.com/pycharm | ~1 GB | Community Edition is enough |
| 3 | **MVTec LOCO AD** | mvtec.com → Research → Datasets → MVTec LOCO AD | 5.7 GB (or 1.0 GB for pushpins alone) | Free for research, registration required |
| 4 | **An API key** | aistudio.google.com (easiest free option) | — | See Part 4 |

Start the dataset download now; it runs while you do the rest.

**Python version matters.** 3.13 may not have wheels for every dependency yet.
3.11 or 3.12 is the safe choice.

---

## Part 2 — Open the project

1. Unzip `keyprompt-ad.zip` somewhere without spaces or accents in the path.
   `C:\projects\keyprompt-ad` is good; `C:\Users\My Name\Desktop\new folder` will
   cause avoidable trouble.
2. PyCharm → **File → Open** → select the `keyprompt-ad` folder (the one
   containing `pyproject.toml`) → **Open**.
3. If PyCharm offers "Create a virtual environment" in a notification bar, you
   can accept it and skip to step 3.4 — otherwise do it manually:

   **File → Settings → Project: keyprompt-ad → Python Interpreter**
   → click the gear icon → **Add** → **Virtualenv Environment** → **New**
   → Base interpreter: your Python 3.11/3.12 → **OK**

4. Wait for the bottom-right progress bar to finish indexing.

**Check:** bottom-right status bar should read something like
`Python 3.12 (keyprompt-ad)`.

---

## Part 3 — Install dependencies

Open PyCharm's terminal: **View → Tool Windows → Terminal** (or `Alt+F12`).
The prompt should already show `(.venv)`. If it doesn't, the interpreter is not
set up — go back to Part 2.

```
pip install -e ".[gemini,dev]"
```

> **On Windows there is no `python3` command.** Use `python`. If you are creating
> the environment by hand rather than letting PyCharm do it:
>
> ```powershell
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> ```
>
> If `python` opens the Microsoft Store, use the launcher instead:
> `py -3.12 -m venv .venv`. If activation reports *"running scripts is
> disabled"*, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first — that
> affects only the current terminal window.

Takes a minute or two.

### Verify before going further

```bash
pytest -q
```

Expected:

```
...................                                          [100%]
19 passed in 2.30s
```

Then:

```bash
python -m keyprompt.cli selftest
```

Expected, ending with:

```
all checks passed
```

**This needs no dataset and no API key.** If it passes, the code is working and
anything that fails later is data, key or model — not the installation. If it
fails, stop here; nothing downstream will work.

---

## Part 4 — Get your API key

1. Go to **aistudio.google.com** → sign in → **Get API key** → **Create API key**.
2. Copy it.
3. In PyCharm's **Project** panel, right-click the `keyprompt-ad` root
   → **New → File** → name it exactly `.env`
4. Paste one line into it:

   ```
   GEMINI_API_KEY=paste_your_key_here
   ```

5. Save (`Ctrl+S`).

The project loads `.env` automatically. You should see
`loaded environment from ...\.env` in the output when you run a command.

**Two warnings:**

- `.env` is already in `.gitignore`. Never remove it from there and never commit
  the file.
- **Rotate the key that was in your original script.** It has been pasted into a
  chat window, so treat it as public. Delete it in AI Studio and create a new one.

---

## Part 5 — Place the dataset

Extract the download so the folders sit exactly like this inside your project:

```
keyprompt-ad/
├── pyproject.toml
├── src/
└── data/
    └── mvtec_loco_anomaly_detection/
        └── pushpins/
            ├── train/good/                    *.png
            ├── validation/good/               *.png
            ├── test/good/                     *.png
            ├── test/logical_anomalies/        *.png
            ├── test/structural_anomalies/     *.png
            ├── ground_truth/logical_anomalies/000/000.png ...
            └── defects_config.json
```

You may need to create the `data` folder yourself. A common mistake is ending up
with a doubled folder such as
`data/mvtec_loco_anomaly_detection/mvtec_loco_anomaly_detection/pushpins`.
Check the path in PyCharm's Project panel.

**Check** in the terminal:

```bash
python -c "import pathlib; p=pathlib.Path('data/mvtec_loco_anomaly_detection/pushpins/train/good'); print(p.exists(), len(list(p.glob('*.png'))) if p.exists() else 0)"
```

Expect `True` followed by a number in the hundreds.

---

## Part 6 — Run configurations

You can type every command in the terminal, but PyCharm run configurations give
you a one-click button and are worth the five minutes.

**Run → Edit Configurations → `+` → Python**

Create this one first:

| Field | Value |
|---|---|
| Name | `1 selftest` |
| **Module name** (switch from "Script path" using the dropdown) | `keyprompt.cli` |
| Parameters | `selftest` |
| Working directory | the `keyprompt-ad` root |

Click **OK**, then the green ▶ button.

Duplicate it (right-click → **Copy Configuration**) for each of these, changing
only the **Name** and **Parameters**:

| Name | Parameters |
|---|---|
| `2 bootstrap pushpins` | `bootstrap --category pushpins --n-images 30` |
| `3 build-prior pushpins` | `build-prior --category pushpins` |
| `4 run pushpins` | `run --category pushpins` |
| `5 evaluate pushpins` | `evaluate --category pushpins` |
| `6 report` | `report` |

Now the whole pipeline is six buttons in the dropdown at the top right.

---

## Part 7 — Run the pipeline

Run configurations `2` → `3`, checking the output of each.

### After `2 bootstrap`

```
consensus over 30 normal images
  surviving slots : 15 {'pushpin': 15}
  dropped clusters: 12 (below support threshold)
```

**`surviving slots` must match the real number of compartments in the tray.**
Open one training image in PyCharm and count them.

If it says `0`, edit configuration `2` and add flags, trying in this order:

```
bootstrap --category pushpins --n-images 30 --invert --overwrite
bootstrap --category pushpins --n-images 30 --invert --min-support 0.5 --overwrite
bootstrap --category pushpins --n-images 30 --invert --min-area 5e-5 --overwrite
```

### After `3 build-prior`

```
built normality graph from 30 shots -> runs/.../pushpins/prior.json
  slots:   15
```

Same check. **A wrong slot count fails silently** — every later number will look
plausible and mean nothing. Do not continue until this is right.

### Smoke test before the full run

Do not press `4` yet. First check the model actually answers, using three images
instead of several hundred. In the terminal:

```bash
python -c "import yaml,pathlib; c=yaml.safe_load(open('configs/default.yaml')); c['limit_per_split']=3; c['categories']=['pushpins']; c['output_root']='runs/smoketest'; pathlib.Path('configs/smoketest.yaml').write_text(yaml.safe_dump(c,sort_keys=False))"

python -m keyprompt.cli --config configs/smoketest.yaml run --category pushpins
python -m keyprompt.cli --config configs/smoketest.yaml evaluate --category pushpins
```

Three things to look for, in order:

1. **No `!` characters** in the run output — those mark API errors.
2. **`parse success` near `1.000`** in the evaluate output.
3. **Scores that differ between images.** If every score is identical, the model
   returned nothing usable and a full run would waste your entire quota.

### Full run

Now press `4 run pushpins`. Expect **10–20 minutes**: the tool deliberately waits
4 seconds between calls to stay inside the free-tier rate limit.

It is resumable and cached. If it stops for any reason, just press ▶ again and it
continues from where it left off.

Then press `5 evaluate pushpins`.

---

## Part 8 — PyCharm-specific gotchas

**The annotation tool opens no window.** `keyprompt annotate` needs an
interactive matplotlib window, and PyCharm's SciView intercepts it.
**File → Settings → Tools → Python Plot** → untick **Show plots in tool window**.
Or just run `keyprompt annotate` from a system terminal instead of PyCharm's.

**"No module named keyprompt".** The editable install did not take. In the
terminal, confirm `(.venv)` is in the prompt, then rerun
`pip install -e ".[gemini,dev]"`.

**Red squiggles under imports that nevertheless run fine.** PyCharm has not
picked up the `src` layout. **File → Settings → Project → Project Structure** →
select the `src` folder → click **Sources** (blue). Cosmetic only.

**`.env` seems ignored.** Look for `loaded environment from ...` in the output.
If it is absent, the file is misnamed — `env.txt` or `.env.txt` are the usual
culprits. Windows Explorer hides extensions; create the file from PyCharm's
Project panel instead.

**Run configuration fails but the terminal works.** The **Working directory**
field is wrong; it must be the `keyprompt-ad` root.

---

## Part 9 — After pushpins works

```
2 bootstrap → 3 build-prior → 4 run → 5 evaluate
```

for `screw_bag`, `splicing_connectors`, `breakfast_box`, `juice_bottle`,
changing `--category` each time. Then `6 report` for the summary table.

`screw_bag` and `juice_bottle` have several component types and the automatic
proposer only handles one, so bootstrap them and then either fix the class names
in `annotations/<category>/*.json` or annotate those two by hand.

Then the ablations, in the terminal:

```bash
python scripts/sweep.py --sweep terms --categories pushpins
```

Run this one first. It includes the `vlm_verdict_only` control, which is the
comparison your paper depends on.

Full details for every step, including a troubleshooting table, are in
`RUNBOOK.md`. The complete annotated source is in `FULL_CODE.md`.
