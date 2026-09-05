# KeyPrompt-AD — complete source listing

Every file in the repository, in dependency order.

---

## Contents

- **Configuration**
  - `src/keyprompt/config.py`
  - `src/keyprompt/dotenv.py`
- **Dataset layer**
  - `src/keyprompt/data/loco.py`
- **Annotation**
  - `src/keyprompt/annotate/schema.py`
  - `src/keyprompt/annotate/tool.py`
  - `src/keyprompt/annotate/auto.py`
- **Normality prior**
  - `src/keyprompt/prior/geometry.py`
  - `src/keyprompt/prior/graph.py`
- **Prompting**
  - `src/keyprompt/prompting/schema.py`
  - `src/keyprompt/prompting/builder.py`
- **VLM providers**
  - `src/keyprompt/providers/base.py`
  - `src/keyprompt/providers/backends.py`
- **Pipeline**
  - `src/keyprompt/pipeline/scoring.py`
  - `src/keyprompt/pipeline/detector.py`
- **Evaluation**
  - `src/keyprompt/eval/metrics.py`
- **Visualisation**
  - `src/keyprompt/viz.py`
- **Command line interface**
  - `src/keyprompt/cli.py`
- **Scripts**
  - `scripts/sweep.py`
  - `scripts/make_paper_assets.py`
  - `scripts/make_method_figures.py`
  - `scripts/git_init.sh`
- **Tests**
  - `tests/test_pipeline.py`
- **Configuration files**
  - `configs/default.yaml`
  - `configs/smoke.yaml`
  - `configs/openrouter.yaml`
  - `configs/categories/pushpins.yaml`
  - `configs/categories/screw_bag.yaml`
  - `configs/categories/splicing_connectors.yaml`
  - `configs/categories/breakfast_box.yaml`
  - `configs/categories/juice_bottle.yaml`
- **Packaging**
  - `pyproject.toml`
  - `requirements.txt`
  - `.env.example`
  - `.gitignore`

---

# Configuration

## `src/keyprompt/config.py`

_138 lines_

```python
"""Configuration objects for the KeyPrompt-AD pipeline.

Everything the pipeline needs is expressed as a dataclass so that a run can be
serialised alongside its results. Reproducibility of a benchmark run means
being able to point at one YAML file and one commit hash.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProviderConfig:
    """Which vision-language model to call, and how."""

    name: str = "gemini"  # gemini | openrouter | groq | echo
    model: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_retries: int = 4
    retry_base_delay: float = 4.0
    # Free tiers are aggressively rate limited. Pace requests rather than
    # burning the quota on 429s.
    min_seconds_between_calls: float = 4.0
    timeout: float = 120.0


@dataclass
class ImageConfig:
    """How reference and query images are prepared before they are sent."""

    max_side: int = 512
    keep_aspect: bool = True
    jpeg_quality: int = 90


@dataclass
class PriorConfig:
    """Parameters for building the normality graph from the reference shots."""

    n_shots: int = 4
    # Reference annotations are aligned to a canonical frame before averaging.
    align: bool = True
    # Minimum positional sigma (normalised units) so that a class observed in
    # only a handful of shots does not collapse to a zero-variance slot.
    min_sigma: float = 0.012


@dataclass
class ScoringConfig:
    """Fusion weights for the continuous anomaly score.

    The VLM emits a discrete verdict, which alone cannot produce an ROC curve.
    The geometric layer converts the returned keypoints into a continuous
    deviation score, so classification threshold and detection quality can be
    separated cleanly.
    """

    # Gating radius (normalised units) for assigning a detection to a prior slot.
    match_radius: float = 0.10
    w_missing: float = 1.0     # prior slots with no detection nearby
    w_extra: float = 0.5       # detections with no prior slot nearby
    w_displacement: float = 1.0  # normalised offset of matched slots
    w_edge: float = 0.8        # violation of learned pairwise distances
    w_vlm: float = 0.6         # the model's own verdict/confidence
    # Displacement and edge terms are expressed in units of the learned sigma;
    # this caps the influence of one wild outlier.
    sigma_clip: float = 6.0


@dataclass
class CategoryConfig:
    """Per-category description used to build the context prompt."""

    name: str = "pushpins"
    # Human-written statement of what a normal part looks like. This is the
    # only piece of domain knowledge the pipeline needs.
    normality_statement: str = ""
    component_classes: List[str] = field(default_factory=list)
    # Defect modes that must be ignored (surface-level variation, etc).
    ignored_variation: List[str] = field(default_factory=list)
    grouping_description: str = ""


@dataclass
class RunConfig:
    dataset_root: Path = Path("data/mvtec_loco_anomaly_detection")
    output_root: Path = Path("runs")
    # Annotations are input data, not run output. They live outside output_root
    # so that an ablation sweep, which varies output_root per variant, reuses
    # one annotation set instead of demanding a fresh one for every run.
    annotations_root: Path = Path("annotations")
    cache_root: Path = Path(".cache/vlm")
    categories: List[str] = field(default_factory=lambda: ["pushpins"])
    seed: int = 0
    limit_per_split: Optional[int] = None  # useful for smoke tests
    use_cache: bool = True

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "RunConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return RunConfig(
            dataset_root=Path(raw.get("dataset_root", "data/mvtec_loco_anomaly_detection")),
            output_root=Path(raw.get("output_root", "runs")),
            annotations_root=Path(raw.get("annotations_root", "annotations")),
            cache_root=Path(raw.get("cache_root", ".cache/vlm")),
            categories=raw.get("categories", ["pushpins"]),
            seed=int(raw.get("seed", 0)),
            limit_per_split=raw.get("limit_per_split"),
            use_cache=bool(raw.get("use_cache", True)),
            provider=ProviderConfig(**raw.get("provider", {})),
            image=ImageConfig(**raw.get("image", {})),
            prior=PriorConfig(**raw.get("prior", {})),
            scoring=ScoringConfig(**raw.get("scoring", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("dataset_root", "output_root", "cache_root", "annotations_root"):
            d[k] = str(d[k])
        return d


def load_category(path: str | Path) -> CategoryConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return CategoryConfig(**raw)

```

## `src/keyprompt/dotenv.py`

_72 lines_

```python
"""Load key/value pairs from a .env file into the process environment.

Written by hand rather than pulled in as a dependency: the format we need is a
dozen lines of parsing, and every extra install step is one more thing that can
fail on someone else's machine.

Existing environment variables always win. That way a value exported in a shell,
or injected by a CI runner or an IDE run configuration, is never silently
overridden by a stale file on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def find_dotenv(start: Optional[Path] = None, filename: str = ".env") -> Optional[Path]:
    """Search for a .env file in this directory and its parents."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> Dict[str, str]:
    """Parse the small subset of the format that matters.

    Supports comments, blank lines, an optional ``export`` prefix, and single or
    double quoted values. Anything more exotic belongs in a real shell script.
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Strip a trailing inline comment only when unquoted.
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        out[key] = value
    return out


def load_dotenv(path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """Load a .env file into ``os.environ``. Returns the file used, if any."""
    p = path or find_dotenv()
    if p is None or not p.is_file():
        return None
    try:
        values = parse_dotenv(p.read_text(encoding="utf-8"))
    except OSError:
        return None
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return p

```

# Dataset layer

## `src/keyprompt/data/loco.py`

_153 lines_

```python
"""Reader for the MVTec LOCO AD directory layout.

Expected on-disk structure (as distributed by MVTec):

    <root>/<category>/
        train/good/*.png
        validation/good/*.png
        test/good/*.png
        test/logical_anomalies/*.png
        test/structural_anomalies/*.png
        ground_truth/logical_anomalies/<stem>/000.png, 001.png, ...
        ground_truth/structural_anomalies/<stem>/...
        defects_config.json

Ground truth is stored as one binary mask per defect *region*, not one mask per
image, which matters when scoring localisation: an image with two missing
pushpins has two separate region masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

ANOMALY_SUBSETS = ["logical_anomalies", "structural_anomalies"]


@dataclass
class Sample:
    image_path: Path
    category: str
    split: str          # train | validation | test
    subset: str         # good | logical_anomalies | structural_anomalies
    label: int          # 0 = normal, 1 = anomalous
    gt_mask_dir: Optional[Path] = None

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def uid(self) -> str:
        return f"{self.category}/{self.split}/{self.subset}/{self.stem}"

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def load_gt_regions(self) -> List[np.ndarray]:
        """Return one boolean mask per annotated defect region."""
        if self.gt_mask_dir is None or not self.gt_mask_dir.is_dir():
            return []
        regions = []
        for p in sorted(self.gt_mask_dir.glob("*.png")):
            m = np.array(Image.open(p).convert("L"))
            regions.append(m > 0)
        return regions

    def load_gt_union(self) -> Optional[np.ndarray]:
        regions = self.load_gt_regions()
        if not regions:
            return None
        out = regions[0].copy()
        for r in regions[1:]:
            out |= r
        return out


@dataclass
class LocoCategory:
    root: Path
    name: str
    defects_config: Dict = field(default_factory=dict)

    @staticmethod
    def open(dataset_root: str | Path, category: str) -> "LocoCategory":
        root = Path(dataset_root) / category
        if not root.is_dir():
            raise FileNotFoundError(
                f"Category directory not found: {root}\n"
                "Download MVTec LOCO AD and point --dataset-root at the "
                "extracted 'mvtec_loco_anomaly_detection' folder."
            )
        cfg_path = root / "defects_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return LocoCategory(root=root, name=category, defects_config=cfg)

    # -- splits ---------------------------------------------------------

    def _images(self, split: str, subset: str) -> List[Path]:
        d = self.root / split / subset
        if not d.is_dir():
            return []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: List[Path] = []
        for e in exts:
            files.extend(d.glob(e))
        return sorted(files)

    def train_normal(self) -> List[Sample]:
        return [
            Sample(p, self.name, "train", "good", 0)
            for p in self._images("train", "good")
        ]

    def validation_normal(self) -> List[Sample]:
        return [
            Sample(p, self.name, "validation", "good", 0)
            for p in self._images("validation", "good")
        ]

    def test(self, limit_per_subset: Optional[int] = None) -> List[Sample]:
        out: List[Sample] = []
        for p in self._images("test", "good")[:limit_per_subset]:
            out.append(Sample(p, self.name, "test", "good", 0))
        for subset in ANOMALY_SUBSETS:
            for p in self._images("test", subset)[:limit_per_subset]:
                out.append(
                    Sample(
                        image_path=p,
                        category=self.name,
                        split="test",
                        subset=subset,
                        label=1,
                        gt_mask_dir=self.root / "ground_truth" / subset / p.stem,
                    )
                )
        return out

    def reference_shots(self, n: int, seed: int = 0) -> List[Sample]:
        """Deterministically sample n normal images to serve as the few-shot set.

        Drawn from the train split. The selection is seeded and recorded so a
        reported number can be traced back to the exact images that produced it.
        """
        pool = self.train_normal()
        if len(pool) < n:
            raise ValueError(f"{self.name}: only {len(pool)} train images, need {n}")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in sorted(idx)]

```

# Annotation

## `src/keyprompt/annotate/schema.py`

_65 lines_

```python
"""Annotation format for reference (OK) images.

A reference annotation is deliberately sparse: a handful of labelled points and
an optional group id per point. This is what a line engineer can produce in a
couple of minutes per part, and it is the entire supervision signal the method
receives.

Coordinates are normalised to [0, 1] with the origin at the top-left corner, so
annotations survive any resizing applied before inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Keypoint:
    cls: str                      # component class, e.g. "pushpin"
    x: float                      # normalised, [0, 1]
    y: float                      # normalised, [0, 1]
    group: Optional[str] = None   # e.g. "compartment_A"; enables shape checks
    note: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ImageAnnotation:
    image_uid: str
    category: str
    width: int
    height: int
    keypoints: List[Keypoint] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k in self.keypoints:
            out[k.cls] = out.get(k.cls, 0) + 1
        return out

    def by_class(self, cls: str) -> List[Keypoint]:
        return [k for k in self.keypoints if k.cls == cls]

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "ImageAnnotation":
        raw = json.loads(Path(path).read_text())
        kps = [Keypoint(**k) for k in raw.pop("keypoints", [])]
        return ImageAnnotation(keypoints=kps, **raw)


def load_annotation_set(directory: str | Path) -> List[ImageAnnotation]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return [ImageAnnotation.from_json(p) for p in sorted(d.glob("*.json"))]

```

## `src/keyprompt/annotate/tool.py`

_114 lines_

```python
"""Minimal click-to-annotate tool for the reference shots.

Usage is intentionally low-tech: matplotlib window, left-click to drop a point,
number keys to switch the active component class, 'g' to start a new group,
'u' to undo, 'w' to write and advance.

Annotating four images per category takes a few minutes, which is the whole
point of the method: no bounding-box campaign, no labelled defects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from .schema import ImageAnnotation, Keypoint

_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


class KeypointAnnotator:
    def __init__(self, classes: List[str], out_dir: str | Path):
        if not classes:
            raise ValueError("at least one component class is required")
        self.classes = classes
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def annotate(
        self,
        image_path: str | Path,
        image_uid: str,
        category: str,
        use_groups: bool = True,
    ) -> Optional[ImageAnnotation]:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(image_path))
        h, w = img.shape[0], img.shape[1]
        ann = ImageAnnotation(image_uid=image_uid, category=category, width=w, height=h)

        state = {"cls_idx": 0, "group": 0, "done": False, "saved": False}
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(img)
        ax.set_axis_off()
        artists: List = []

        def title() -> None:
            ax.set_title(
                f"[{Path(image_path).name}]  class={self.classes[state['cls_idx']]}  "
                f"group={state['group'] if use_groups else '-'}  n={len(ann.keypoints)}\n"
                "click=add   1..9=class   g=new group   u=undo   w=write+close   q=skip",
                fontsize=9,
            )
            fig.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not ax or event.xdata is None:
                return
            cls = self.classes[state["cls_idx"]]
            kp = Keypoint(
                cls=cls,
                x=round(float(event.xdata) / w, 5),
                y=round(float(event.ydata) / h, 5),
                group=f"g{state['group']}" if use_groups else None,
            )
            ann.keypoints.append(kp)
            color = _PALETTE[state["cls_idx"] % len(_PALETTE)]
            dot = ax.plot(event.xdata, event.ydata, "o", ms=9, mfc=color, mec="white")[0]
            txt = ax.annotate(
                f"{cls[:3]}{state['group'] if use_groups else ''}",
                (event.xdata, event.ydata),
                textcoords="offset points",
                xytext=(8, 6),
                color=color,
                fontsize=8,
            )
            artists.append((dot, txt))
            title()

        def on_key(event) -> None:
            if event.key in [str(i) for i in range(1, 10)]:
                i = int(event.key) - 1
                if i < len(self.classes):
                    state["cls_idx"] = i
            elif event.key == "g":
                state["group"] += 1
            elif event.key == "u" and ann.keypoints:
                ann.keypoints.pop()
                dot, txt = artists.pop()
                dot.remove()
                txt.remove()
            elif event.key == "w":
                state["saved"] = True
                plt.close(fig)
                return
            elif event.key == "q":
                plt.close(fig)
                return
            title()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        title()
        plt.show()

        if not state["saved"] or not ann.keypoints:
            return None
        out = self.out_dir / f"{Path(image_path).stem}.json"
        ann.to_json(out)
        print(f"wrote {out}  ({len(ann.keypoints)} keypoints)")
        return ann

```

## `src/keyprompt/annotate/auto.py`

_322 lines_

```python
"""Deriving reference annotations without hand-clicking every point.

Manual annotation is the weakest link in the pipeline, not because it is slow
but because four hand-placed points are a poor sample from which to estimate a
positional tolerance. The consensus route below is cheaper *and* statistically
better: propose points on many normal training images, align them into a common
frame, and keep only the clusters that recur across most of them.

Two proposers are provided.

``propose_blobs``
    Classical connected-component detection on a thresholded image. No model,
    no network, deterministic. Works well when components are visually distinct
    from the background (pushpins in a tray, screws on a light surface) and
    poorly otherwise. Reach for this first.

``propose_vlm``
    Asks the vision-language model to locate components from the text normality
    statement alone, with no coordinates supplied. Fully hands-off, and the
    obvious thing to try -- but note the failure mode: the prior is built from
    the model's own detections and then used to grade that same model. Errors
    are absorbed rather than exposed. Report results from this route separately
    from the manually-annotated ones; do not silently mix them.

Neither proposer knows which cluster is which component class when several
classes are present. For multi-class categories (screw_bag, juice_bottle) the
practical route is proposal followed by a quick human pass to assign labels,
which is still far less work than placing every point by hand.

Nothing here ever reads ``ground_truth/``. Those masks exist only for anomalous
test images; using them to build the normality prior would be test-set leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..annotate.schema import ImageAnnotation, Keypoint
from ..prior.geometry import align_by_class, apply_transform, match_points

Proposal = Tuple[str, float, float]  # (class, x, y) in normalised coordinates
Proposer = Callable[[Image.Image], List[Proposal]]


# -- proposers -----------------------------------------------------------


def propose_blobs(
    image: Image.Image,
    cls: str = "component",
    min_area_frac: float = 2e-4,
    max_area_frac: float = 5e-2,
    invert: bool = False,
    threshold: Optional[float] = None,
    max_points: int = 64,
) -> List[Proposal]:
    """Connected-component proposal on a thresholded grayscale image.

    ``threshold`` defaults to Otsu's method. Set ``invert`` when components are
    darker than the background. Area bounds are expressed as a fraction of the
    image so the same settings survive a change of resolution.
    """
    from scipy import ndimage

    g = np.asarray(image.convert("L"), dtype=float) / 255.0
    thr = _otsu(g) if threshold is None else threshold
    mask = (g < thr) if invert else (g > thr)

    # Close small gaps so a specular highlight does not split one component.
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

    labels, n = ndimage.label(mask)
    if n == 0:
        return []

    H, W = g.shape
    area = H * W
    lo, hi = min_area_frac * area, max_area_frac * area
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if lo <= s <= hi]
    if not keep:
        return []

    centroids = ndimage.center_of_mass(mask, labels, index=keep)
    ranked = sorted(zip(keep, centroids), key=lambda kv: -sizes[kv[0] - 1])[:max_points]
    return [(cls, float(c[1] / W), float(c[0] / H)) for _, c in ranked]


def propose_vlm(
    image: Image.Image,
    provider,
    category_cfg,
    image_cfg,
) -> List[Proposal]:
    """Ask the VLM to locate components from the text description alone.

    No coordinates are supplied, so this does not depend on any prior existing
    yet. It is the bootstrap step for a fully hands-off run.
    """
    from ..prompting.schema import parse_response
    from ..providers.base import prepare_image

    classes = category_cfg.component_classes or ["component"]
    prompt = f"""
This image shows a correct, non-defective example of '{category_cfg.name}'.

{category_cfg.normality_statement.strip()}

Locate every instance of these component types: {", ".join(classes)}.

Report the centre of each one in normalised coordinates, where x runs from 0 at
the left edge to 1 at the right edge and y from 0 at the top to 1 at the bottom.
Do not omit any instance, and do not invent instances that are not visible.

Reply with a single JSON object and nothing else:
{{"verdict": "OK", "confidence": 0.0,
  "detected": [{{"cls": "<one of: {', '.join(classes)}>", "x": <float>, "y": <float>}}],
  "missing": [], "reasoning": ""}}
""".strip()

    resp = provider.generate([prompt, prepare_image(image, image_cfg)])
    if resp.error:
        return []
    try:
        parsed = parse_response(resp.text)
    except Exception:  # noqa: BLE001 - a bad proposal is dropped, not fatal
        return []
    return [(d.cls, d.x, d.y) for d in parsed.detected]


# -- consensus -----------------------------------------------------------


@dataclass
class ConsensusReport:
    n_images: int
    n_slots: int
    per_class: Dict[str, int]
    dropped_clusters: int
    mean_support: float
    mean_spread: float

    def describe(self) -> str:
        return (
            f"consensus over {self.n_images} normal images\n"
            f"  surviving slots : {self.n_slots} {self.per_class}\n"
            f"  dropped clusters: {self.dropped_clusters} (below support threshold)\n"
            f"  mean support    : {self.mean_support:.2f} of images\n"
            f"  mean spread     : {self.mean_spread:.4f} (normalised units)"
        )


def consensus_annotations(
    images: Sequence[Image.Image],
    proposer: Proposer,
    min_support: float = 0.6,
    cluster_radius: float = 0.05,
    align: bool = True,
) -> Tuple[List[ImageAnnotation], ConsensusReport]:
    """Turn noisy per-image proposals into clean reference annotations.

    A cluster of points that recurs at the same place across most normal images
    is a real component slot. One that appears in a handful is a detection
    artefact. That distinction is the whole filter, and it is what makes 30
    automatic proposals more trustworthy than 4 careful manual clicks.

    Returns annotations in the same format the manual tool produces, so the
    downstream graph builder is unchanged.
    """
    if not images:
        raise ValueError("no images supplied")

    raw: List[Dict[str, np.ndarray]] = []
    for img in images:
        by_cls: Dict[str, List[List[float]]] = {}
        for cls, x, y in proposer(img):
            by_cls.setdefault(cls, []).append([x, y])
        raw.append({c: np.array(v, dtype=float).reshape(-1, 2) for c, v in by_cls.items()})

    # Anchor on the image with the median number of proposals: an outlier frame
    # where the detector over- or under-fired should not define the frame.
    totals = [sum(len(p) for p in r.values()) for r in raw]
    anchor_idx = int(np.argsort(totals)[len(totals) // 2])
    anchor = raw[anchor_idx]

    aligned: List[Dict[str, np.ndarray]] = []
    for i, r in enumerate(raw):
        if not align or i == anchor_idx:
            aligned.append(r)
            continue
        fit = align_by_class(r, anchor, radius=cluster_radius * 2.0)
        if fit is None:
            aligned.append(r)
        else:
            R, s, t = fit
            aligned.append({c: apply_transform(p, R, s, t) for c, p in r.items()})

    classes = sorted({c for a in aligned for c in a})
    slots: Dict[str, np.ndarray] = {}
    supports: Dict[str, np.ndarray] = {}
    spreads: List[float] = []
    dropped = 0

    for cls in classes:
        pooled, owners = [], []
        for i, a in enumerate(aligned):
            for p in a.get(cls, np.zeros((0, 2))):
                pooled.append(p)
                owners.append(i)
        if not pooled:
            continue

        centres, members = _cluster(np.array(pooled), cluster_radius)
        keep_centres, keep_support = [], []
        for c, idxs in zip(centres, members):
            support = len({owners[j] for j in idxs}) / len(images)
            if support >= min_support:
                keep_centres.append(c)
                keep_support.append(support)
                pts = np.array(pooled)[idxs]
                spreads.append(float(pts.std(axis=0).mean()))
            else:
                dropped += 1
        if keep_centres:
            slots[cls] = np.array(keep_centres)
            supports[cls] = np.array(keep_support)

    # Emit one annotation per source image, keeping only proposals that joined a
    # surviving cluster, expressed back in that image's own coordinates.
    annotations: List[ImageAnnotation] = []
    for i, (img, r) in enumerate(zip(images, raw)):
        kps: List[Keypoint] = []
        for cls, centres in slots.items():
            pts = aligned[i].get(cls, np.zeros((0, 2)))
            if len(pts) == 0:
                continue
            pairs, _, _ = match_points(pts, centres, cluster_radius)
            for pi, si in pairs:
                orig = r[cls][pi]  # untransformed, matching this image
                kps.append(
                    Keypoint(cls=cls, x=float(orig[0]), y=float(orig[1]), group=f"{cls}_{si}")
                )
        annotations.append(
            ImageAnnotation(
                image_uid=f"auto/{i}",
                category="auto",
                width=img.width,
                height=img.height,
                keypoints=kps,
            )
        )

    report = ConsensusReport(
        n_images=len(images),
        n_slots=sum(len(v) for v in slots.values()),
        per_class={c: len(v) for c, v in slots.items()},
        dropped_clusters=dropped,
        mean_support=float(np.mean([s for v in supports.values() for s in v])) if supports else 0.0,
        mean_spread=float(np.mean(spreads)) if spreads else 0.0,
    )
    return annotations, report


# -- helpers -------------------------------------------------------------


def _cluster(points: np.ndarray, radius: float) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Greedy agglomerative clustering, densest seed first.

    Simple by design: the point sets are small and the geometry is already
    aligned, so a heavier algorithm buys nothing but another dependency.
    """
    n = len(points)
    if n == 0:
        return [], []
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    density = (d <= radius).sum(axis=1)
    order = np.argsort(-density)

    unassigned = set(range(n))
    centres: List[np.ndarray] = []
    members: List[List[int]] = []
    for seed in order:
        if seed not in unassigned:
            continue
        group = [j for j in unassigned if d[seed, j] <= radius]
        unassigned -= set(group)
        centres.append(points[group].mean(axis=0))
        members.append(group)
    return centres, members


def _otsu(g: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(g.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b = np.nan_to_num(sigma_b)

    # On a cleanly separated image, between-class variance is flat across the
    # whole empty valley between the two modes. Taking argmax would put the
    # threshold on the first tied bin, i.e. hard against the darker cluster,
    # where any noise splits those pixels arbitrarily. Take the middle of the
    # tied range instead so the threshold sits in the gap.
    best = sigma_b.max()
    if best <= 0:
        return 0.5
    tied = np.flatnonzero(sigma_b >= best - 1e-12)
    return float(mids[int(round((tied[0] + tied[-1]) / 2.0))])

```

# Normality prior

## `src/keyprompt/prior/geometry.py`

_122 lines_

```python
"""Geometric primitives shared by the prior builder and the scorer.

Two operations carry most of the weight:

1. A similarity fit (rotation, uniform scale, translation) between two labelled
   point sets. Parts are rarely presented in a perfectly repeatable pose, and
   without alignment a two-pixel jig shift would masquerade as a defect.

2. A gated one-to-one assignment between predicted points and prior slots.
   Greedy nearest-neighbour matching produces order-dependent results, so the
   assignment is solved optimally and then gated by a radius, which lets points
   remain unmatched rather than forcing a bad pairing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def similarity_transform(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Umeyama's closed-form solution. Returns (R, s, t) such that
    ``s * src @ R.T + t`` approximates ``dst``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be matching (N, 2) arrays")
    n = src.shape[0]
    if n == 0:
        return np.eye(2), 1.0, np.zeros(2)
    if n == 1:
        return np.eye(2), 1.0, dst[0] - src[0]

    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return R, s, t


def apply_transform(pts: np.ndarray, R: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    return s * (pts @ R.T) + t


def match_points(
    pred: np.ndarray,
    prior: np.ndarray,
    radius: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Optimal one-to-one assignment of predictions to prior slots.

    Returns ``(pairs, unmatched_pred, unmatched_prior)`` where ``pairs`` holds
    (pred_index, prior_index) tuples whose separation is within ``radius``.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    prior = np.asarray(prior, dtype=float).reshape(-1, 2)
    if len(pred) == 0:
        return [], [], list(range(len(prior)))
    if len(prior) == 0:
        return [], list(range(len(pred))), []

    cost = np.linalg.norm(pred[:, None, :] - prior[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(cost)

    pairs: List[Tuple[int, int]] = []
    used_pred, used_prior = set(), set()
    for i, j in zip(ri, ci):
        if cost[i, j] <= radius:
            pairs.append((int(i), int(j)))
            used_pred.add(int(i))
            used_prior.add(int(j))
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    unmatched_prior = [j for j in range(len(prior)) if j not in used_prior]
    return pairs, unmatched_pred, unmatched_prior


def align_by_class(
    pred_by_class: dict[str, np.ndarray],
    prior_by_class: dict[str, np.ndarray],
    radius: float,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Estimate a global pose correction from confidently matched points.

    A first pass matches without alignment to find a consensus set; the
    transform is then fitted on that set only, so missing or spurious points do
    not drag the pose estimate.
    """
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    for cls, prior_pts in prior_by_class.items():
        pred_pts = pred_by_class.get(cls)
        if pred_pts is None or len(pred_pts) == 0 or len(prior_pts) == 0:
            continue
        pairs, _, _ = match_points(pred_pts, prior_pts, radius)
        for pi, qi in pairs:
            src_pts.append(pred_pts[pi])
            dst_pts.append(prior_pts[qi])
    if len(src_pts) < 2:
        return None
    return similarity_transform(np.array(src_pts), np.array(dst_pts))


def pairwise_distances(pts: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

```

## `src/keyprompt/prior/graph.py`

_301 lines_

```python
"""The normality graph: a compact statistical model of what "OK" looks like.

Built from K annotated reference images (K is typically 4). It stores, per
component class:

* the expected number of instances,
* canonical slot positions in a common frame,
* per-slot positional spread,

plus the pairwise distances between slots, with their spread. The edge
statistics are what let the system flag a part whose components are all present
but spaced wrongly, which is the failure mode that pure presence-checking
misses.

The graph plays two roles. It is injected into the prompt as an explicit layout
specification, and it is used afterwards to convert the model's returned points
into a continuous deviation score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..annotate.schema import ImageAnnotation
from .geometry import (
    align_by_class,
    apply_transform,
    match_points,
    similarity_transform,
)


@dataclass
class ClassPrior:
    cls: str
    expected_count: int
    slots: np.ndarray            # (M, 2) canonical positions, normalised
    slot_sigma: np.ndarray       # (M,) positional spread per slot
    groups: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cls": self.cls,
            "expected_count": self.expected_count,
            "slots": self.slots.round(4).tolist(),
            "slot_sigma": self.slot_sigma.round(4).tolist(),
            "groups": self.groups,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ClassPrior":
        return ClassPrior(
            cls=d["cls"],
            expected_count=int(d["expected_count"]),
            slots=np.array(d["slots"], dtype=float).reshape(-1, 2),
            slot_sigma=np.array(d["slot_sigma"], dtype=float).reshape(-1),
            groups=d.get("groups", []),
        )


@dataclass
class NormalityGraph:
    category: str
    n_shots: int
    classes: Dict[str, ClassPrior] = field(default_factory=dict)
    # Flattened slot table so edges can index across classes.
    slot_index: List[Tuple[str, int]] = field(default_factory=list)
    edge_mean: Optional[np.ndarray] = None   # (S, S)
    edge_sigma: Optional[np.ndarray] = None  # (S, S)
    source_uids: List[str] = field(default_factory=list)

    # -- construction ---------------------------------------------------

    @staticmethod
    def build(
        annotations: List[ImageAnnotation],
        category: str,
        align: bool = True,
        min_sigma: float = 0.012,
    ) -> "NormalityGraph":
        if not annotations:
            raise ValueError("no reference annotations supplied")

        # Anchor on the most complete reference, not simply the first. If the
        # anchor is missing a component, that slot is absent from the canonical
        # frame and no later shot can reintroduce it -- a silent, permanent hole
        # in the prior. This matters most for automatically proposed
        # annotations, where any single frame may have dropped a detection.
        anchor_idx = int(np.argmax([len(a.keypoints) for a in annotations]))
        ref = annotations[anchor_idx]
        ordered = [ref] + [a for i, a in enumerate(annotations) if i != anchor_idx]
        ref_by_cls = _points_by_class(ref)

        aligned: List[Dict[str, np.ndarray]] = [ref_by_cls]
        for ann in ordered[1:]:
            cur = _points_by_class(ann)
            if align:
                fit = align_by_class(cur, ref_by_cls, radius=0.12)
                if fit is not None:
                    R, s, t = fit
                    cur = {c: apply_transform(p, R, s, t) for c, p in cur.items()}
            aligned.append(cur)

        classes: Dict[str, ClassPrior] = {}
        all_names = sorted({c for a in aligned for c in a})
        for cls in all_names:
            anchor = ref_by_cls.get(cls, np.zeros((0, 2)))
            if len(anchor) == 0:
                # Class absent from the anchor shot: fall back to the first
                # reference that contains it.
                for a in aligned:
                    if len(a.get(cls, [])) > 0:
                        anchor = a[cls]
                        break
            m = len(anchor)
            if m == 0:
                continue

            stacks = [[] for _ in range(m)]
            counts: List[int] = []
            for a in aligned:
                pts = a.get(cls, np.zeros((0, 2)))
                counts.append(len(pts))
                if len(pts) == 0:
                    continue
                pairs, _, _ = match_points(pts, anchor, radius=0.15)
                for pi, si in pairs:
                    stacks[si].append(pts[pi])

            slots = np.zeros((m, 2))
            sigma = np.zeros(m)
            for i in range(m):
                if stacks[i]:
                    arr = np.array(stacks[i])
                    slots[i] = arr.mean(0)
                    sigma[i] = max(float(arr.std(0).mean()), min_sigma)
                else:
                    slots[i] = anchor[i]
                    sigma[i] = min_sigma * 2.0

            groups = _groups_for(ref, cls, m)
            classes[cls] = ClassPrior(
                cls=cls,
                expected_count=int(round(float(np.median(counts)))),
                slots=slots,
                slot_sigma=sigma,
                groups=groups,
            )

        g = NormalityGraph(
            category=category,
            n_shots=len(annotations),
            classes=classes,
            source_uids=[a.image_uid for a in ordered],
        )
        g._build_edges(aligned)
        return g

    def _build_edges(self, aligned: List[Dict[str, np.ndarray]]) -> None:
        self.slot_index = [
            (cls, i) for cls in sorted(self.classes) for i in range(len(self.classes[cls].slots))
        ]
        S = len(self.slot_index)
        if S < 2:
            self.edge_mean = np.zeros((0, 0))
            self.edge_sigma = np.zeros((0, 0))
            return

        per_shot: List[np.ndarray] = []
        for a in aligned:
            coords = np.full((S, 2), np.nan)
            for cls in sorted(self.classes):
                pts = a.get(cls, np.zeros((0, 2)))
                if len(pts) == 0:
                    continue
                pairs, _, _ = match_points(pts, self.classes[cls].slots, radius=0.15)
                base = self.slot_index.index((cls, 0))
                for pi, si in pairs:
                    coords[base + si] = pts[pi]
            per_shot.append(coords)

        D = np.full((len(per_shot), S, S), np.nan)
        for k, coords in enumerate(per_shot):
            diff = coords[:, None, :] - coords[None, :, :]
            D[k] = np.linalg.norm(diff, axis=-1)

        with np.errstate(invalid="ignore"):
            self.edge_mean = np.nanmean(D, axis=0)
            self.edge_sigma = np.nanstd(D, axis=0)
        self.edge_mean = np.nan_to_num(self.edge_mean, nan=0.0)
        self.edge_sigma = np.nan_to_num(self.edge_sigma, nan=0.0)
        # Distances measured from few shots are optimistically tight; floor the
        # spread so a single-pixel wobble is not scored as a five-sigma event.
        self.edge_sigma = np.maximum(self.edge_sigma, 0.015)

    # -- accessors ------------------------------------------------------

    def expected_counts(self) -> Dict[str, int]:
        return {c: p.expected_count for c, p in self.classes.items()}

    def slots_by_class(self) -> Dict[str, np.ndarray]:
        return {c: p.slots for c, p in self.classes.items()}

    def total_slots(self) -> int:
        return sum(len(p.slots) for p in self.classes.values())

    def describe(self, max_neighbours: int = 3) -> str:
        """Plain-language layout specification injected into the prompt.

        The full edge set grows quadratically with the number of slots, which
        would dominate the prompt for a category like pushpins. Only the
        ``max_neighbours`` closest spacings per slot are listed: they carry
        almost all of the constraint, since a component displaced far enough to
        break a long-range distance will already have broken a short-range one.
        """
        lines: List[str] = []
        for cls in sorted(self.classes):
            p = self.classes[cls]
            lines.append(f"- {cls}: expected count = {p.expected_count}")
            for i, (xy, sg) in enumerate(zip(p.slots, p.slot_sigma)):
                grp = p.groups[i] if i < len(p.groups) and p.groups[i] else "-"
                lines.append(
                    f"    slot {cls}[{i}] at (x={xy[0]:.3f}, y={xy[1]:.3f}) "
                    f"tolerance ~{sg:.3f}, group {grp}"
                )
        if self.edge_mean is not None and len(self.slot_index) >= 2:
            shown: set[tuple[int, int]] = set()
            for a in range(len(self.slot_index)):
                row = self.edge_mean[a].copy()
                row[a] = np.inf
                row[row <= 0] = np.inf
                order = np.argsort(row)[:max_neighbours]
                for b in order:
                    if not np.isfinite(row[b]):
                        continue
                    shown.add((min(a, int(b)), max(a, int(b))))
            if shown:
                lines.append("- characteristic spacings between neighbouring slots (normalised):")
                for a, b in sorted(shown):
                    ca, ia = self.slot_index[a]
                    cb, ib = self.slot_index[b]
                    lines.append(
                        f"    {ca}[{ia}] - {cb}[{ib}] : {self.edge_mean[a, b]:.3f} "
                        f"(+/- {self.edge_sigma[a, b]:.3f})"
                    )
        return "\n".join(lines)

    # -- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "category": self.category,
                    "n_shots": self.n_shots,
                    "classes": {c: cp.to_dict() for c, cp in self.classes.items()},
                    "slot_index": self.slot_index,
                    "edge_mean": None if self.edge_mean is None else self.edge_mean.round(4).tolist(),
                    "edge_sigma": None if self.edge_sigma is None else self.edge_sigma.round(4).tolist(),
                    "source_uids": self.source_uids,
                },
                indent=2,
            )
        )

    @staticmethod
    def load(path: str | Path) -> "NormalityGraph":
        d = json.loads(Path(path).read_text())
        return NormalityGraph(
            category=d["category"],
            n_shots=int(d["n_shots"]),
            classes={c: ClassPrior.from_dict(v) for c, v in d["classes"].items()},
            slot_index=[tuple(x) for x in d.get("slot_index", [])],
            edge_mean=None if d.get("edge_mean") is None else np.array(d["edge_mean"], dtype=float),
            edge_sigma=None if d.get("edge_sigma") is None else np.array(d["edge_sigma"], dtype=float),
            source_uids=d.get("source_uids", []),
        )


# -- helpers -------------------------------------------------------------


def _points_by_class(ann: ImageAnnotation) -> Dict[str, np.ndarray]:
    out: Dict[str, List[List[float]]] = {}
    for k in ann.keypoints:
        out.setdefault(k.cls, []).append([k.x, k.y])
    return {c: np.array(v, dtype=float).reshape(-1, 2) for c, v in out.items()}


def _groups_for(ann: ImageAnnotation, cls: str, m: int) -> List[Optional[str]]:
    g = [k.group for k in ann.keypoints if k.cls == cls]
    if len(g) < m:
        g = g + [None] * (m - len(g))
    return g[:m]

```

# Prompting

## `src/keyprompt/prompting/schema.py`

_131 lines_

```python
"""Response schema for the VLM.

Free-form text answers are unusable for benchmarking: parsing them is brittle
and the failure modes are silent. Every provider is therefore asked for JSON
matching this schema, and responses that do not parse are recorded as explicit
failures rather than quietly dropped.

Models drift from the requested shape in predictable ways -- code fences, a
0-10 confidence scale, pixel coordinates, "classification" instead of
"verdict". The parser absorbs those variations rather than discarding an
otherwise usable answer, because silently dropping responses would bias the
benchmark toward whichever model happens to be the most obedient formatter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field


@dataclass
class DetectedComponent:
    """One component the model claims to have located."""

    cls: str                      # class name, exactly as given in the layout
    x: float                      # normalised horizontal position in [0,1]
    y: float                      # normalised vertical position in [0,1]
    slot: Optional[str] = None    # matching slot id, when the model identifies one


@dataclass
class InspectionResult:
    verdict: str = "OK"                      # OK or NOT OK
    confidence: float = 0.5                  # 0 = certainly OK, 1 = certainly defective
    detected: List[DetectedComponent] = field(default_factory=list)
    missing: List[DetectedComponent] = field(default_factory=list)
    reasoning: str = ""


JSON_SCHEMA_HINT = """
{
  "verdict": "OK" | "NOT OK",
  "confidence": <float between 0 and 1>,
  "detected": [{"cls": "<class>", "x": <float>, "y": <float>, "slot": "<class>[i]" }],
  "missing":  [{"cls": "<class>", "x": <float>, "y": <float>, "slot": "<class>[i]" }],
  "reasoning": "<one or two sentences>"
}
""".strip()


def parse_response(text: str) -> InspectionResult:
    """Recover an InspectionResult from a model response.

    Providers differ in how faithfully they honour a JSON instruction, so the
    parser tolerates code fences and leading prose before giving up.
    """
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in response: {cleaned[:200]!r}")
        payload = json.loads(cleaned[start : end + 1])

    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    d = _coerce(payload)
    return InspectionResult(
        verdict=d["verdict"],
        confidence=d["confidence"],
        detected=[DetectedComponent(**c) for c in d["detected"]],
        missing=[DetectedComponent(**c) for c in d["missing"]],
        reasoning=d["reasoning"],
    )


def _coerce(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the small variations models produce around field naming."""
    out = dict(d)
    for alias in ("classification", "label", "result", "status"):
        if "verdict" not in out and alias in out:
            out["verdict"] = out[alias]
    out.setdefault("verdict", "OK")
    out["verdict"] = "NOT OK" if "NOT" in str(out["verdict"]).upper() else "OK"

    conf = out.get("confidence", out.get("anomaly_score", 0.5))
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.5
    # Some models answer on a 0-10 scale despite the instruction.
    out["confidence"] = conf / 10.0 if conf > 1.0 else max(0.0, conf)

    for key in ("detected", "missing"):
        items = out.get(key) or []
        fixed: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cls = it.get("cls") or it.get("class") or it.get("type") or "unknown"
            if "x" in it and "y" in it:
                x, y = it["x"], it["y"]
            elif isinstance(it.get("position"), (list, tuple)) and len(it["position"]) >= 2:
                x, y = it["position"][0], it["position"][1]
            else:
                continue
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue
            # Guard against models that answer in pixels or on a 0-1000 grid.
            if x > 1.5 or y > 1.5:
                scale = 1000.0 if max(x, y) <= 1000.0 else max(x, y)
                x, y = x / scale, y / scale
            fixed.append({"cls": str(cls), "x": x, "y": y, "slot": it.get("slot")})
        out[key] = fixed

    out["reasoning"] = str(out.get("reasoning") or out.get("defect_description") or "")[:600]
    return out

```

## `src/keyprompt/prompting/builder.py`

_111 lines_

```python
"""Prompt assembly.

The prompt has three parts, in this order:

1. A task frame describing the inspection job in general terms.
2. The layout specification, generated automatically from the normality graph.
   This is the piece that makes the method transferable: change the reference
   annotations and the specification rewrites itself.
3. The few-shot block, pairing each reference image with its exact coordinate
   annotation, followed by the query image.

Nothing here is hand-tuned per category beyond a short normality statement,
which keeps the comparison across the five LOCO categories honest.
"""

from __future__ import annotations

from typing import List, Tuple

from ..annotate.schema import ImageAnnotation
from ..config import CategoryConfig
from ..prior.graph import NormalityGraph
from .schema import JSON_SCHEMA_HINT

TASK_FRAME = """
You are inspecting images of a manufactured assembly photographed in a fixed
setup. Your job is to check that every required component is present and sitting
where it belongs, and to report the result as structured data.

Work from geometry, not appearance. A part is defective when a component is
absent, duplicated, or sitting outside its tolerance, or when the spacing
between components departs from the reference layout. A part is NOT defective
because of scratches, stains, smudges, reflections, dust, or minor differences
in lighting or shade. Ignore that kind of surface variation entirely.

All coordinates are normalised: x runs from 0 at the left edge to 1 at the
right edge, y runs from 0 at the top edge to 1 at the bottom edge.
""".strip()


def build_context_prompt(category: CategoryConfig, graph: NormalityGraph) -> str:
    parts: List[str] = [TASK_FRAME, ""]

    parts.append(f"COMPONENT TYPES for '{category.name}':")
    classes = category.component_classes or sorted(graph.classes)
    for c in classes:
        parts.append(f"- {c}")
    parts.append("")

    if category.normality_statement:
        parts.append("WHAT A CORRECT PART LOOKS LIKE:")
        parts.append(category.normality_statement.strip())
        parts.append("")

    if category.grouping_description:
        parts.append("HOW COMPONENTS ARE GROUPED:")
        parts.append(category.grouping_description.strip())
        parts.append("")

    parts.append(
        f"REFERENCE LAYOUT, derived from {graph.n_shots} correct examples. "
        "Treat the tolerances as guidance rather than hard limits:"
    )
    parts.append(graph.describe())
    parts.append("")

    if category.ignored_variation:
        parts.append("DO NOT report any of the following as defects:")
        for v in category.ignored_variation:
            parts.append(f"- {v}")
        parts.append("")

    parts.append(
        "A part is NOT OK if any of the following hold: a component listed above "
        "is absent; an extra component appears where the layout expects none; a "
        "component sits well outside its tolerance; or the spacing between two "
        "components departs clearly from the characteristic value."
    )
    return "\n".join(parts)


def build_shot_block(index: int, annotation: ImageAnnotation) -> str:
    lines = [f"--- Reference example {index} (correct part) ---"]
    by_cls: dict[str, List[Tuple[float, float]]] = {}
    for k in annotation.keypoints:
        by_cls.setdefault(k.cls, []).append((k.x, k.y))
    for cls in sorted(by_cls):
        coords = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in by_cls[cls])
        lines.append(f"{cls} ({len(by_cls[cls])}): {coords}")
    lines.append(f"verdict: OK")
    lines.append(f"--- end reference example {index} ---")
    return "\n".join(lines)


def build_query_prompt() -> str:
    return f"""
Now inspect the final image.

Report every component you can locate, using the same class names and the same
normalised coordinate convention as the reference examples. If a component the
layout requires is absent, list it under "missing" with the position where it
should have been.

Set "confidence" to your probability that the part is defective: near 0 when
the part is clearly correct, near 1 when a component is clearly absent or
badly out of place.

Reply with a single JSON object and nothing else, in exactly this shape:

{JSON_SCHEMA_HINT}
""".strip()

```

# VLM providers

## `src/keyprompt/providers/base.py`

_174 lines_

```python
"""Provider abstraction over vision-language APIs.

Every backend receives the same interleaved sequence of text and images and
returns raw text. Keeping the interface this thin means swapping Gemini for a
Qwen checkpoint on OpenRouter is a one-line config change, which is what makes
the cross-model ablation in the evaluation cheap to run.

Three concerns live here rather than in the individual backends:

* a content-addressed response cache, so a rerun costs nothing and results stay
  reproducible after the free tier resets;
* retry with exponential backoff, because free tiers return 429 constantly;
* a request pacer, for the same reason.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..config import ImageConfig, ProviderConfig

Content = Union[str, Image.Image]


@dataclass
class VLMResponse:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def prepare_image(img: Image.Image, cfg: ImageConfig) -> Image.Image:
    """Downscale before upload.

    Resolution is the dominant cost and latency driver. It is also a genuine
    experimental variable: how small can the input get before the localisation
    degrades? That ablation is reported in the evaluation.
    """
    out = img.convert("RGB")
    if cfg.keep_aspect:
        out.thumbnail((cfg.max_side, cfg.max_side), Image.Resampling.LANCZOS)
    else:
        out = out.resize((cfg.max_side, cfg.max_side), Image.Resampling.LANCZOS)
    return out


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLMProvider(ABC):
    """Base class handling caching, pacing and retries."""

    def __init__(self, cfg: ProviderConfig, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_at = 0.0

    # -- subclass hook --------------------------------------------------

    @abstractmethod
    def _call(self, contents: List[Content]) -> VLMResponse:
        """Issue one request. Raise on failure; retries are handled upstream."""

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"environment variable {self.cfg.api_key_env} is not set.\n"
                f"Fix: copy .env.example to .env in the repository root and add "
                f"a line reading  {self.cfg.api_key_env}=<your key>\n"
                f"The .env file is loaded automatically and is gitignored."
            )
        return key

    # -- public API -----------------------------------------------------

    def generate(self, contents: List[Content]) -> VLMResponse:
        key = self._cache_key(contents)
        cached = self._cache_read(key)
        if cached is not None:
            return cached

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            self._pace()
            try:
                resp = self._call(contents)
                self._cache_write(key, resp)
                return resp
            except Exception as exc:  # noqa: BLE001 - surfaced in the response
                last_err = exc
                delay = self.cfg.retry_base_delay * (2 ** attempt)
                if attempt < self.cfg.max_retries - 1:
                    print(f"  [{self.cfg.name}] attempt {attempt + 1} failed ({exc}); "
                          f"retrying in {delay:.0f}s")
                    time.sleep(delay)

        return VLMResponse(text="", latency_s=0.0, error=str(last_err))

    # -- internals ------------------------------------------------------

    def _pace(self) -> None:
        gap = time.time() - self._last_call_at
        wait = self.cfg.min_seconds_between_calls - gap
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.time()

    def _cache_key(self, contents: List[Content]) -> str:
        h = hashlib.sha256()
        h.update(f"{self.cfg.name}|{self.cfg.model}|{self.cfg.temperature}".encode())
        for c in contents:
            if isinstance(c, str):
                h.update(b"T")
                h.update(c.encode())
            else:
                h.update(b"I")
                buf = io.BytesIO()
                c.save(buf, format="PNG")
                h.update(hashlib.sha256(buf.getvalue()).digest())
        return h.hexdigest()[:32]

    def _cache_read(self, key: str) -> Optional[VLMResponse]:
        if not (self.use_cache and self.cache_dir):
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return VLMResponse(
            text=d["text"],
            latency_s=d.get("latency_s", 0.0),
            prompt_tokens=d.get("prompt_tokens"),
            output_tokens=d.get("output_tokens"),
            cached=True,
            error=d.get("error"),
        )

    def _cache_write(self, key: str, resp: VLMResponse) -> None:
        if not (self.use_cache and self.cache_dir) or resp.error:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(
                {
                    "text": resp.text,
                    "latency_s": resp.latency_s,
                    "prompt_tokens": resp.prompt_tokens,
                    "output_tokens": resp.output_tokens,
                },
                indent=2,
            )
        )

```

## `src/keyprompt/providers/backends.py`

_151 lines_

```python
"""Concrete backends.

All four are usable at zero cost at the time of writing, though free-tier terms
move around; check current limits before planning a full sweep.

  gemini      Google AI Studio. Native structured output, generous free tier,
              strong at spatial grounding. Default choice.
  openrouter  Aggregator exposing several open-weight VLMs on a free tier,
              e.g. Qwen2.5-VL and Llama vision variants. Useful for showing the
              method is not tied to one proprietary model.
  groq        Fast inference on open-weight vision models; the lowest-latency
              option, which matters for the throughput argument.
  echo        Offline stub. Returns a fixed empty detection so the pipeline,
              scorer and metrics can be exercised without network access.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from PIL import Image

from ..prompting.schema import JSON_SCHEMA_HINT
from .base import Content, VLMProvider, VLMResponse, image_to_b64


class GeminiProvider(VLMProvider):
    def _call(self, contents: List[Content]) -> VLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        t0 = time.time()
        resp = client.models.generate_content(
            model=self.cfg.model,
            contents=list(contents),
            config=types.GenerateContentConfig(
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        dt = time.time() - t0
        usage = getattr(resp, "usage_metadata", None)
        return VLMResponse(
            text=resp.text or "",
            latency_s=dt,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )


class _OpenAICompatProvider(VLMProvider):
    """Shared implementation for OpenAI-style chat endpoints."""

    base_url = ""
    extra_headers: Dict[str, str] = {}

    def _call(self, contents: List[Content]) -> VLMResponse:
        import requests

        parts: List[Dict[str, Any]] = []
        for c in contents:
            if isinstance(c, str):
                parts.append({"type": "text", "text": c})
            elif isinstance(c, Image.Image):
                b64 = image_to_b64(c)
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )

        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        t0 = time.time()
        r = requests.post(
            self.base_url, headers=headers, data=json.dumps(body), timeout=self.cfg.timeout
        )
        dt = time.time() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        payload = r.json()
        usage = payload.get("usage", {})
        return VLMResponse(
            text=payload["choices"][0]["message"]["content"] or "",
            latency_s=dt,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class OpenRouterProvider(_OpenAICompatProvider):
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    extra_headers = {"X-Title": "KeyPrompt-AD"}


class GroqProvider(_OpenAICompatProvider):
    base_url = "https://api.groq.com/openai/v1/chat/completions"


class EchoProvider(VLMProvider):
    """Offline stub for smoke tests and CI."""

    @property
    def api_key(self) -> str:  # no key needed
        return "offline"

    def _call(self, contents: List[Content]) -> VLMResponse:
        time.sleep(0.01)
        return VLMResponse(
            text=json.dumps(
                {"verdict": "OK", "confidence": 0.0, "detected": [], "missing": [],
                 "reasoning": "offline stub response"}
            ),
            latency_s=0.01,
            prompt_tokens=0,
            output_tokens=0,
        )


_REGISTRY = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "echo": EchoProvider,
}


def build_provider(cfg, cache_dir=None, use_cache: bool = True) -> VLMProvider:
    name = cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown provider '{cfg.name}'. Options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, cache_dir=cache_dir, use_cache=use_cache)


__all__ = ["build_provider", "JSON_SCHEMA_HINT"]

```

# Pipeline

## `src/keyprompt/pipeline/scoring.py`

_180 lines_

```python
"""Turning a language-model verdict into a number you can plot an ROC curve on.

A VLM naturally emits a categorical answer ("NOT OK") plus a list of points.
Categorical answers give you a single operating point and nothing else, which
makes them awkward to compare against unsupervised baselines that report
AUROC. The scorer closes that gap: it audits the returned points against the
normality graph and produces a continuous deviation score.

Four terms are combined:

* missing   - prior slots with no nearby detection
* extra     - detections not explained by any slot
* displace  - offset of matched slots, in units of the learned tolerance
* edge      - violation of the learned pairwise spacings

The model's own verdict enters as a fifth, deliberately small term. The
geometry does the discriminating; the verdict mostly breaks ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ScoringConfig
from ..prior.geometry import align_by_class, apply_transform, match_points
from ..prior.graph import NormalityGraph


@dataclass
class ScoreBreakdown:
    score: float
    missing: float
    extra: float
    displacement: float
    edge: float
    vlm: float
    n_missing: int
    n_extra: int
    missing_points: List[Tuple[float, float]] = field(default_factory=list)
    extra_points: List[Tuple[float, float]] = field(default_factory=list)
    displaced_points: List[Tuple[float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def defect_points(self) -> List[Tuple[float, float]]:
        """Points to overlay on the image as the localisation output."""
        return self.missing_points + self.extra_points + self.displaced_points

    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 5),
            "terms": {
                "missing": round(self.missing, 5),
                "extra": round(self.extra, 5),
                "displacement": round(self.displacement, 5),
                "edge": round(self.edge, 5),
                "vlm": round(self.vlm, 5),
            },
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "missing_points": [[round(x, 4), round(y, 4)] for x, y in self.missing_points],
            "extra_points": [[round(x, 4), round(y, 4)] for x, y in self.extra_points],
            "displaced_points": [[round(x, 4), round(y, 4)] for x, y in self.displaced_points],
            "notes": self.notes,
        }


def score_detection(
    detected: Dict[str, List[Tuple[float, float]]],
    graph: NormalityGraph,
    cfg: ScoringConfig,
    vlm_verdict: Optional[str] = None,
    vlm_confidence: Optional[float] = None,
) -> ScoreBreakdown:
    """Compare a set of detected component positions against the prior."""
    pred = {c: np.asarray(v, dtype=float).reshape(-1, 2) for c, v in detected.items()}
    prior = graph.slots_by_class()

    # Global pose correction so that a shifted jig is not read as a defect.
    fit = align_by_class(pred, prior, radius=cfg.match_radius * 1.5)
    if fit is not None:
        R, s, t = fit
        pred = {c: apply_transform(p, R, s, t) for c, p in pred.items()}

    total_slots = max(graph.total_slots(), 1)
    n_missing = n_extra = 0
    disp_terms: List[float] = []
    missing_pts: List[Tuple[float, float]] = []
    extra_pts: List[Tuple[float, float]] = []
    displaced_pts: List[Tuple[float, float]] = []
    notes: List[str] = []

    matched_coords: Dict[Tuple[str, int], np.ndarray] = {}

    for cls, cp in graph.classes.items():
        pts = pred.get(cls, np.zeros((0, 2)))
        pairs, un_pred, un_prior = match_points(pts, cp.slots, cfg.match_radius)

        for pi, si in pairs:
            d = float(np.linalg.norm(pts[pi] - cp.slots[si]))
            z = min(d / max(cp.slot_sigma[si], 1e-6), cfg.sigma_clip)
            disp_terms.append(z / cfg.sigma_clip)
            matched_coords[(cls, si)] = pts[pi]
            if z > 3.0:
                displaced_pts.append((float(pts[pi][0]), float(pts[pi][1])))
                notes.append(f"{cls}[{si}] displaced by {d:.3f} ({z:.1f} sigma)")

        for si in un_prior:
            n_missing += 1
            missing_pts.append((float(cp.slots[si][0]), float(cp.slots[si][1])))
            notes.append(f"{cls}[{si}] not observed at expected location")

        for pi in un_pred:
            n_extra += 1
            extra_pts.append((float(pts[pi][0]), float(pts[pi][1])))
            notes.append(f"unexpected {cls} at ({pts[pi][0]:.3f}, {pts[pi][1]:.3f})")

    missing_term = n_missing / total_slots
    extra_term = n_extra / total_slots
    disp_term = float(np.mean(disp_terms)) if disp_terms else 0.0
    edge_term = _edge_violation(matched_coords, graph, cfg)

    vlm_term = 0.0
    if vlm_confidence is not None:
        vlm_term = float(np.clip(vlm_confidence, 0.0, 1.0))
    elif vlm_verdict is not None:
        vlm_term = 1.0 if str(vlm_verdict).strip().upper().startswith("NOT") else 0.0

    raw = (
        cfg.w_missing * missing_term
        + cfg.w_extra * extra_term
        + cfg.w_displacement * disp_term
        + cfg.w_edge * edge_term
        + cfg.w_vlm * vlm_term
    )
    denom = cfg.w_missing + cfg.w_extra + cfg.w_displacement + cfg.w_edge + cfg.w_vlm
    score = float(raw / denom) if denom > 0 else 0.0

    return ScoreBreakdown(
        score=score,
        missing=missing_term,
        extra=extra_term,
        displacement=disp_term,
        edge=edge_term,
        vlm=vlm_term,
        n_missing=n_missing,
        n_extra=n_extra,
        missing_points=missing_pts,
        extra_points=extra_pts,
        displaced_points=displaced_pts,
        notes=notes,
    )


def _edge_violation(
    matched: Dict[Tuple[str, int], np.ndarray],
    graph: NormalityGraph,
    cfg: ScoringConfig,
) -> float:
    """Mean normalised deviation of observed spacings from learned spacings."""
    if graph.edge_mean is None or len(matched) < 2 or not graph.slot_index:
        return 0.0
    keys = [k for k in matched if k in graph.slot_index]
    if len(keys) < 2:
        return 0.0

    zs: List[float] = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia = graph.slot_index.index(keys[a])
            ib = graph.slot_index.index(keys[b])
            mu = graph.edge_mean[ia, ib]
            sd = max(graph.edge_sigma[ia, ib], 1e-6)
            if mu <= 0:
                continue
            d = float(np.linalg.norm(matched[keys[a]] - matched[keys[b]]))
            zs.append(min(abs(d - mu) / sd, cfg.sigma_clip) / cfg.sigma_clip)
    return float(np.mean(zs)) if zs else 0.0

```

## `src/keyprompt/pipeline/detector.py`

_169 lines_

```python
"""End-to-end inspection of a single image.

The flow per query image is: assemble prompt -> one VLM call -> parse JSON ->
audit the returned points against the normality graph -> emit a continuous
score and a set of defect coordinates.

There is no training step anywhere in this file, which is the point. The only
per-category cost is annotating K reference images once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..annotate.schema import ImageAnnotation
from ..config import CategoryConfig, RunConfig
from ..data.loco import Sample
from ..prior.graph import NormalityGraph
from ..prompting.builder import build_context_prompt, build_query_prompt, build_shot_block
from ..prompting.schema import InspectionResult, parse_response
from ..providers.base import Content, VLMProvider, prepare_image
from .scoring import ScoreBreakdown, score_detection


@dataclass
class Prediction:
    uid: str
    category: str
    subset: str
    label: int
    score: float
    verdict: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    parse_ok: bool = True
    error: Optional[str] = None
    breakdown: Optional[Dict] = None
    defect_points: List[Tuple[float, float]] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "uid": self.uid,
            "category": self.category,
            "subset": self.subset,
            "label": self.label,
            "score": round(self.score, 5),
            "verdict": self.verdict,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cached": self.cached,
            "parse_ok": self.parse_ok,
            "error": self.error,
            "breakdown": self.breakdown,
            "defect_points": [[round(x, 4), round(y, 4)] for x, y in self.defect_points],
            "reasoning": self.reasoning,
        }


class KeyPromptDetector:
    def __init__(
        self,
        cfg: RunConfig,
        category: CategoryConfig,
        graph: NormalityGraph,
        reference_images: List[Image.Image],
        reference_annotations: List[ImageAnnotation],
        provider: VLMProvider,
    ):
        if len(reference_images) != len(reference_annotations):
            raise ValueError("reference images and annotations must align one-to-one")
        self.cfg = cfg
        self.category = category
        self.graph = graph
        self.provider = provider
        self.ref_images = [prepare_image(im, cfg.image) for im in reference_images]
        self.ref_annotations = reference_annotations
        self._context = build_context_prompt(category, graph)
        self._query = build_query_prompt()

    # -- prompt ---------------------------------------------------------

    def _contents(self, query_image: Image.Image) -> List[Content]:
        contents: List[Content] = [self._context, ""]
        for i, (img, ann) in enumerate(zip(self.ref_images, self.ref_annotations), start=1):
            contents.append(build_shot_block(i, ann))
            contents.append(img)
        contents.append(self._query)
        contents.append(prepare_image(query_image, self.cfg.image))
        return contents

    # -- inference ------------------------------------------------------

    def predict(self, sample: Sample) -> Prediction:
        t0 = time.time()
        img = sample.load_image()
        resp = self.provider.generate(self._contents(img))

        if resp.error:
            return Prediction(
                uid=sample.uid,
                category=sample.category,
                subset=sample.subset,
                label=sample.label,
                score=0.5,  # neutral: an API failure is not evidence either way
                verdict="ERROR",
                latency_s=time.time() - t0,
                cached=resp.cached,
                parse_ok=False,
                error=resp.error,
            )

        try:
            parsed: InspectionResult = parse_response(resp.text)
            parse_ok = True
            err = None
        except Exception as exc:  # noqa: BLE001
            parsed = InspectionResult(verdict="OK", confidence=0.5)
            parse_ok = False
            err = f"parse failure: {exc}"

        detected: Dict[str, List[Tuple[float, float]]] = {}
        for d in parsed.detected:
            detected.setdefault(d.cls, []).append((d.x, d.y))

        breakdown: ScoreBreakdown = score_detection(
            detected=detected,
            graph=self.graph,
            cfg=self.cfg.scoring,
            vlm_verdict=parsed.verdict,
            vlm_confidence=parsed.confidence,
        )

        # Points the model explicitly reported as missing are localisation
        # evidence in their own right and are merged with the geometric ones.
        pts = list(breakdown.defect_points()) + [(m.x, m.y) for m in parsed.missing]

        return Prediction(
            uid=sample.uid,
            category=sample.category,
            subset=sample.subset,
            label=sample.label,
            score=breakdown.score,
            verdict=parsed.verdict,
            latency_s=resp.latency_s if resp.latency_s else time.time() - t0,
            prompt_tokens=resp.prompt_tokens,
            output_tokens=resp.output_tokens,
            cached=resp.cached,
            parse_ok=parse_ok,
            error=err,
            breakdown=breakdown.to_dict(),
            defect_points=_dedupe(pts),
            reasoning=parsed.reasoning,
        )


def _dedupe(points: List[Tuple[float, float]], tol: float = 0.01) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        if all(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol for q in out):
            out.append(p)
    return out

```

# Evaluation

## `src/keyprompt/eval/metrics.py`

_226 lines_

```python
"""Metrics.

Three families are reported, because a paper that only reports AUROC invites
the obvious question of whether the system is usable on a line.

Detection
    Image-level AUROC and average precision, plus the best achievable F1 and
    the accuracy at that operating point. AUROC is also broken out for the
    logical and structural test subsets separately, since MVTec LOCO
    deliberately separates the two and the method is aimed squarely at the
    logical half.

Localisation
    The official LOCO localisation metric (saturated per-region overlap)
    assumes dense masks; this system emits points, so a point-based analogue is
    used instead. Region recall asks what fraction of annotated defect regions
    contain at least one predicted point. Point precision asks what fraction of
    predicted points land inside some annotated region. Both are reported over
    anomalous test images only, and the choice is stated rather than smuggled
    in as if it were the official protocol.

Efficiency
    Latency percentiles and token counts, measured on uncached calls only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -- detection -----------------------------------------------------------


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUROC via the rank statistic, with correct handling of ties."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def best_f1(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    """Maximum F1 over all thresholds, and the threshold that achieves it."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    k = int(np.argmax(f1))
    return float(f1[k]), float(ss[k])


def classification_at(labels: Sequence[int], scores: Sequence[float], thr: float) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


# -- localisation --------------------------------------------------------


@dataclass
class LocalisationCase:
    points: List[Tuple[float, float]]   # normalised predicted defect points
    regions: List[np.ndarray]           # boolean GT masks, image resolution


def localisation_scores(cases: List[LocalisationCase], tolerance_px: int = 12) -> Dict[str, float]:
    """Point-based localisation.

    A predicted point counts as hitting a region if it falls inside the mask, or
    within ``tolerance_px`` of it. The tolerance matters for "missing component"
    defects, where the annotated region marks an empty slot and the natural
    prediction sits at its centre but need not land exactly inside.
    """
    hit_regions = total_regions = 0
    hit_points = total_points = 0
    images_localised = images_with_regions = 0

    for case in cases:
        if not case.regions:
            continue
        images_with_regions += 1
        H, W = case.regions[0].shape
        px = [(int(round(x * W)), int(round(y * H))) for x, y in case.points]
        dil = [_dilate(r, tolerance_px) for r in case.regions]

        any_hit = False
        for r in dil:
            total_regions += 1
            if any(_inside(r, u, v) for u, v in px):
                hit_regions += 1
                any_hit = True
        for u, v in px:
            total_points += 1
            if any(_inside(r, u, v) for r in dil):
                hit_points += 1
        if any_hit:
            images_localised += 1

    return {
        "region_recall": hit_regions / total_regions if total_regions else float("nan"),
        "point_precision": hit_points / total_points if total_points else float("nan"),
        "image_hit_rate": images_localised / images_with_regions if images_with_regions else float("nan"),
        "n_regions": total_regions,
        "n_points": total_points,
    }


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

        st = iterate_structure(generate_binary_structure(2, 1), r)
        return binary_dilation(mask, structure=st)
    except ImportError:
        return mask


def _inside(mask: np.ndarray, u: int, v: int) -> bool:
    h, w = mask.shape
    return 0 <= v < h and 0 <= u < w and bool(mask[v, u])


# -- aggregation ---------------------------------------------------------


def summarise(
    labels: Sequence[int],
    scores: Sequence[float],
    subsets: Sequence[str],
    latencies: Sequence[float],
    prompt_tokens: Sequence[Optional[int]] = (),
    output_tokens: Sequence[Optional[int]] = (),
    parse_ok: Sequence[bool] = (),
) -> Dict[str, object]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    sub = np.asarray(subsets)

    f1, thr = best_f1(y, s)
    out: Dict[str, object] = {
        "n": int(len(y)),
        "n_normal": int((y == 0).sum()),
        "n_anomalous": int((y == 1).sum()),
        "auroc": roc_auc(y, s),
        "average_precision": average_precision(y, s),
        "f1_max": f1,
        "operating_point": classification_at(y, s, thr),
    }

    normal_mask = sub == "good"
    for name in ("logical_anomalies", "structural_anomalies"):
        m = normal_mask | (sub == name)
        if (sub == name).sum() > 0:
            out[f"auroc_{name.replace('_anomalies', '')}"] = roc_auc(y[m], s[m])

    lat = np.asarray([x for x in latencies if x and x > 0], dtype=float)
    if lat.size:
        out["latency"] = {
            "mean_s": float(lat.mean()),
            "p50_s": float(np.percentile(lat, 50)),
            "p95_s": float(np.percentile(lat, 95)),
            "max_s": float(lat.max()),
        }
    pt = [x for x in prompt_tokens if x]
    ot = [x for x in output_tokens if x]
    if pt or ot:
        out["tokens"] = {
            "prompt_mean": float(np.mean(pt)) if pt else None,
            "output_mean": float(np.mean(ot)) if ot else None,
        }
    if len(parse_ok):
        out["parse_success_rate"] = float(np.mean([1.0 if x else 0.0 for x in parse_ok]))
    return out

```

# Visualisation

## `src/keyprompt/viz.py`

_83 lines_

```python
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

```

# Command line interface

## `src/keyprompt/cli.py`

_512 lines_

```python
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

```

# Scripts

## `scripts/sweep.py`

_137 lines_

```python
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

```

## `scripts/make_paper_assets.py`

_282 lines_

```python
#!/usr/bin/env python3
"""Generate the manuscript's figures and result tables from real run outputs.

This closes the loop between the pipeline and the paper. Everything it emits is
derived from files the pipeline wrote; nothing is invented. If a run is missing,
the corresponding table cell stays as a visible placeholder rather than being
quietly filled with a plausible-looking number.

Usage, from the KeyPrompt-AD repository root:

    python make_paper_assets.py --runs runs/gemini-flash-4shot \\
                                --sweep runs/sweep/summary.json \\
                                --out paper/

Produces:
    fig_shots.pdf         logical AUROC against reference-shot count
    fig_qualitative.png   contact sheet of predictions over ground truth
    table_main.tex        Table 1, per-category detection
    table_ablation.tex    Table 3, ablations
    table_localisation.tex Table 4, localisation and parse rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

CATEGORIES = [
    ("pushpins", "Pushpins"),
    ("screw_bag", "Screw bag"),
    ("splicing_connectors", "Splicing connectors"),
    ("breakfast_box", "Breakfast box"),
    ("juice_bottle", "Juice bottle"),
]

MISSING = r"\nd"  # renders as the red placeholder defined in the manuscript


def cell(value: Optional[float], places: int = 3) -> str:
    """Format a metric, or emit the visible placeholder if it is absent."""
    if value is None:
        return MISSING
    try:
        f = float(value)
    except (TypeError, ValueError):
        return MISSING
    if f != f:  # NaN
        return MISSING
    return f"{f:.{places}f}"


def load_metrics(runs: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key, _ in CATEGORIES:
        p = runs / key / "metrics.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


# -- tables --------------------------------------------------------------


def table_main(metrics: Dict[str, dict]) -> str:
    rows: List[str] = []
    aurocs: List[float] = []
    for key, label in CATEGORIES:
        m = metrics.get(key, {})
        if isinstance(m.get("auroc"), (int, float)):
            aurocs.append(m["auroc"])
        rows.append(
            f"{label:<20} & {cell(m.get('auroc'))} & {cell(m.get('auroc_logical'))} "
            f"& {cell(m.get('auroc_structural'))} & {cell(m.get('average_precision'))} "
            f"& {cell(m.get('f1_max'))} \\\\"
        )
    mean = f"{sum(aurocs)/len(aurocs):.3f}" if len(aurocs) == len(CATEGORIES) else MISSING
    body = "\n".join(rows)
    return f"""% Generated by make_paper_assets.py -- do not edit by hand.
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
Category & AUROC & AUROC$_{{\\mathrm{{log}}}}$ & AUROC$_{{\\mathrm{{str}}}}$ & AP & $F_1^{{\\max}}$ \\\\
\\midrule
{body}
\\midrule
Mean                 & {mean} & {MISSING} & {MISSING} & {MISSING} & {MISSING} \\\\
\\botrule
\\end{{tabular}}
"""


def table_localisation(metrics: Dict[str, dict]) -> str:
    rows = []
    for key, label in CATEGORIES:
        m = metrics.get(key, {})
        loc = m.get("localisation") or {}
        rows.append(
            f"{label:<20} & {cell(loc.get('region_recall'))} "
            f"& {cell(loc.get('point_precision'))} "
            f"& {cell(m.get('parse_success_rate'))} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Generated by make_paper_assets.py -- do not edit by hand.
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Category & Region recall & Point precision & Parse success \\\\
\\midrule
{body}
\\botrule
\\end{{tabular}}
"""


def table_ablation(sweep: Optional[Path]) -> str:
    if not sweep or not sweep.exists():
        return ("% No sweep summary found; run scripts/sweep.py first.\n"
                "% Table 3 in the manuscript retains its placeholders.\n")
    rows_raw = json.loads(sweep.read_text())
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows_raw:
        grouped.setdefault(r["sweep"], {}).setdefault(r["variant"], []).append(r)

    lines = []
    for sweep_name in sorted(grouped):
        variants = grouped[sweep_name]
        lines.append(f"\\multirow{{{len(variants)}}}{{*}}{{{sweep_name}}}")
        for i, variant in enumerate(sorted(variants)):
            entries = variants[variant]
            vals = [e["auroc_logical"] for e in entries
                    if isinstance(e.get("auroc_logical"), (int, float))]
            lats = [e["latency_p95"] for e in entries
                    if isinstance(e.get("latency_p95"), (int, float))]
            auroc = cell(sum(vals) / len(vals) if vals else None)
            lat = cell(sum(lats) / len(lats) if lats else None, places=2)
            prefix = "" if i == 0 else " "
            lines.append(f"{prefix} & {variant.replace('_', ' ')} & {auroc} & {lat} \\\\")
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    return f"""% Generated by make_paper_assets.py -- do not edit by hand.
\\begin{{tabular}}{{@{{}}llcc@{{}}}}
\\toprule
Factor & Setting & AUROC$_{{\\mathrm{{log}}}}$ & Latency (s) \\\\
\\midrule
{body}
\\botrule
\\end{{tabular}}
"""


# -- figures -------------------------------------------------------------


def figure_shots(sweep: Optional[Path], out: Path) -> Optional[Path]:
    """Logical AUROC against reference-shot count, with per-category spread."""
    if not sweep or not sweep.exists():
        print("  skip fig_shots: no sweep summary")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in json.loads(sweep.read_text()) if r["sweep"] == "shots"]
    if not rows:
        print("  skip fig_shots: no shot-count rows")
        return None

    by_k: Dict[int, List[float]] = {}
    for r in rows:
        try:
            k = int(str(r["variant"]).split("_")[-1])
        except ValueError:
            continue
        v = r.get("auroc_logical")
        if isinstance(v, (int, float)) and v == v:
            by_k.setdefault(k, []).append(v)
    if not by_k:
        print("  skip fig_shots: no usable values")
        return None

    ks = sorted(by_k)
    mean = [sum(by_k[k]) / len(by_k[k]) for k in ks]
    lo = [min(by_k[k]) for k in ks]
    hi = [max(by_k[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.fill_between(ks, lo, hi, alpha=0.18, label="range across categories")
    ax.plot(ks, mean, marker="o", linewidth=1.8, label="mean")
    ax.set_xlabel("Reference shots $K$")
    ax.set_ylabel(r"AUROC$_{\mathrm{logical}}$")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out / "fig_shots.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def figure_qualitative(runs: Path, dataset: Optional[Path], out: Path,
                       per_category: int = 2) -> Optional[Path]:
    """Contact sheet of predicted defect coordinates over ground-truth regions."""
    if dataset is None:
        print("  skip fig_qualitative: pass --dataset to render overlays")
        return None
    import sys
    sys.path.insert(0, "src")
    try:
        from PIL import Image
        from keyprompt.data.loco import LocoCategory
        from keyprompt.viz import overlay, save_grid
    except ImportError as exc:
        print(f"  skip fig_qualitative: {exc}")
        return None

    panels: List[Image.Image] = []
    for key, _ in CATEGORIES:
        pred_file = runs / key / "predictions.jsonl"
        if not pred_file.exists():
            continue
        rows = [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]
        anomalous = [r for r in rows if r["label"] == 1 and r.get("defect_points")]
        anomalous.sort(key=lambda r: -r["score"])
        try:
            cat = LocoCategory.open(dataset, key)
        except FileNotFoundError:
            continue
        by_uid = {s.uid: s for s in cat.test()}
        for r in anomalous[:per_category]:
            s = by_uid.get(r["uid"])
            if s is None:
                continue
            panels.append(
                overlay(s.load_image(), breakdown=r.get("breakdown"),
                        gt_mask=s.load_gt_union())
            )
    if not panels:
        print("  skip fig_qualitative: no predictions with defect points")
        return None
    p = out / "fig_qualitative.png"
    save_grid(panels, p, cols=4, cell=360)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="run directory")
    ap.add_argument("--sweep", type=Path, default=None, help="sweep summary.json")
    ap.add_argument("--dataset", type=Path, default=None, help="MVTec LOCO root")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.runs)
    print(f"loaded metrics for {len(metrics)}/{len(CATEGORIES)} categories")
    if len(metrics) < len(CATEGORIES):
        missing = [k for k, _ in CATEGORIES if k not in metrics]
        print(f"  still to run: {', '.join(missing)}")
        print("  those rows will remain visible placeholders, not blanks")

    for name, text in [
        ("table_main.tex", table_main(metrics)),
        ("table_localisation.tex", table_localisation(metrics)),
        ("table_ablation.tex", table_ablation(args.sweep)),
    ]:
        (args.out / name).write_text(text)
        print(f"  wrote {args.out / name}")

    for p in (figure_shots(args.sweep, args.out),
              figure_qualitative(args.runs, args.dataset, args.out)):
        if p:
            print(f"  wrote {p}")

    print("\nPaste the generated tables into the manuscript, replacing the "
          "placeholder tabular blocks, then delete the \\needsdata macro so "
          "any remaining placeholder breaks the build.")


if __name__ == "__main__":
    main()

```

## `scripts/make_method_figures.py`

_238 lines_

```python
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

```

## `scripts/git_init.sh`

_23 lines_

```bash
#!/usr/bin/env bash
# One-time repository setup. Run from the repo root.
set -euo pipefail

if [ -f .env ]; then
  echo "note: .env exists and is gitignored. Confirm it is not staged:"
  echo "      git check-ignore -v .env"
fi

git init -b main
git add .
git status --short

cat <<'MSG'

Review the staged list above. It must NOT contain:
  .env, any *.key, anything under data/, runs/ or .cache/

Then:
  git commit -m "KeyPrompt-AD: keypoint-grounded few-shot logical anomaly detection"
  git remote add origin git@github.com:<you>/keyprompt-ad.git
  git push -u origin main
MSG

```

# Tests

## `tests/test_pipeline.py`

_271 lines_

```python
"""Tests for the parts of the system that do not need an API key.

The geometric layer is where correctness actually matters: if the prior or the
matching is wrong, every reported number is wrong, and the failure is silent.
These run in under a second and need no dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from keyprompt.annotate.schema import ImageAnnotation, Keypoint
from keyprompt.config import ScoringConfig
from keyprompt.eval.metrics import average_precision, best_f1, roc_auc
from keyprompt.pipeline.scoring import score_detection
from keyprompt.prior.geometry import apply_transform, match_points, similarity_transform
from keyprompt.prior.graph import NormalityGraph
from keyprompt.prompting.schema import parse_response

GRID = [(0.12 + 0.19 * c, 0.20 + 0.30 * r) for r in range(3) for c in range(5)]


def _refs(n: int = 4, jitter: float = 0.004, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        kps = [
            Keypoint("pushpin", x + rng.normal(0, jitter), y + rng.normal(0, jitter), group=f"g{i // 5}")
            for i, (x, y) in enumerate(GRID)
        ]
        out.append(ImageAnnotation(f"ref{k}", "pushpins", 1000, 600, kps))
    return out


@pytest.fixture(scope="module")
def graph() -> NormalityGraph:
    return NormalityGraph.build(_refs(), "pushpins")


# -- geometry ------------------------------------------------------------


def test_similarity_transform_recovers_known_pose():
    src = np.array(GRID)
    theta = np.deg2rad(12.0)
    R_true = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = 1.3 * (src @ R_true.T) + np.array([0.05, -0.02])

    R, s, t = similarity_transform(src, dst)
    assert s == pytest.approx(1.3, abs=1e-6)
    assert np.allclose(apply_transform(src, R, s, t), dst, atol=1e-8)


def test_matching_is_gated_by_radius():
    pred = np.array([[0.10, 0.10], [0.90, 0.90]])
    prior = np.array([[0.11, 0.11], [0.50, 0.50]])
    pairs, un_pred, un_prior = match_points(pred, prior, radius=0.05)
    assert pairs == [(0, 0)]
    assert un_pred == [1] and un_prior == [1]


def test_matching_handles_empty_inputs():
    pairs, up, uq = match_points(np.zeros((0, 2)), np.array([[0.1, 0.1]]), 0.1)
    assert pairs == [] and up == [] and uq == [0]


# -- prior ---------------------------------------------------------------


def test_prior_learns_the_right_number_of_slots(graph):
    assert graph.total_slots() == len(GRID)
    assert graph.expected_counts()["pushpin"] == len(GRID)


def test_prior_round_trips_through_disk(graph, tmp_path):
    p = tmp_path / "prior.json"
    graph.save(p)
    back = NormalityGraph.load(p)
    assert back.total_slots() == graph.total_slots()
    assert np.allclose(back.classes["pushpin"].slots, graph.classes["pushpin"].slots, atol=1e-3)


def test_layout_description_is_prompt_ready(graph):
    text = graph.describe()
    assert "expected count = 15" in text
    assert "characteristic spacings" in text
    # the edge list must stay bounded, not quadratic in the slot count
    assert sum(1 for l in text.splitlines() if " - " in l and ":" in l) <= 3 * len(GRID)


# -- scoring -------------------------------------------------------------


def _score(points, graph, verdict="OK", conf=0.0):
    return score_detection({"pushpin": points}, graph, ScoringConfig(), verdict, conf)


def test_otsu_picks_the_middle_of_the_valley():
    """A flat between-class variance must not pin the threshold to one mode."""
    from keyprompt.annotate.auto import _otsu

    g = np.concatenate([np.full(5000, 0.15), np.full(5000, 0.85)])
    thr = _otsu(g)
    assert 0.15 < thr < 0.85
    assert abs(thr - 0.5) < 0.1


def test_defect_modes_score_above_normal(graph):
    rng = np.random.default_rng(7)
    normal = _score([(x + rng.normal(0, 0.004), y + rng.normal(0, 0.004)) for x, y in GRID],
                    graph=graph)
    missing = _score([p for i, p in enumerate(GRID) if i != 7], graph=graph, verdict="NOT OK", conf=0.9)
    shifted = _score([(x + (0.09 if i == 3 else 0.0), y) for i, (x, y) in enumerate(GRID)],
                     graph=graph, verdict="NOT OK", conf=0.8)
    extra = _score(GRID + [(0.55, 0.85)], graph=graph, verdict="NOT OK", conf=0.7)

    assert missing.n_missing == 1
    assert extra.n_extra == 1
    assert shifted.displacement > normal.displacement
    for defective in (missing, shifted, extra):
        assert defective.score > normal.score


def test_global_translation_is_not_a_defect(graph):
    """A shifted jig must not be reported as a fault."""
    shifted_all = [(x + 0.03, y - 0.02) for x, y in GRID]
    b = _score(shifted_all, graph=graph)
    assert b.n_missing == 0 and b.n_extra == 0
    assert b.score < 0.10


def test_missing_point_is_reported_at_the_empty_slot(graph):
    b = _score([p for i, p in enumerate(GRID) if i != 7], graph=graph)
    assert len(b.missing_points) == 1
    assert np.allclose(b.missing_points[0], GRID[7], atol=0.02)


# -- metrics -------------------------------------------------------------


def test_auroc_matches_hand_computed_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties


def test_average_precision_and_f1():
    assert average_precision([0, 1, 1], [0.1, 0.9, 0.8]) == pytest.approx(1.0)
    f1, thr = best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert f1 == pytest.approx(1.0)
    assert thr == pytest.approx(0.8)


# -- response parsing ----------------------------------------------------


def test_parser_survives_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"verdict":"NOT OK","confidence":0.8,' \
          '"detected":[{"cls":"pushpin","x":0.1,"y":0.2}],"missing":[]}\n```'
    r = parse_response(raw)
    assert r.verdict == "NOT OK"
    assert r.detected[0].x == pytest.approx(0.1)


def test_parser_rescales_pixel_and_ten_point_answers():
    raw = '{"classification":"NOT OK","anomaly_score":8,' \
          '"detected":[{"class":"pushpin","position":[500,300]}]}'
    r = parse_response(raw)
    assert r.confidence == pytest.approx(0.8)
    assert 0.0 <= r.detected[0].x <= 1.0


# -- automatic annotation ------------------------------------------------


def _noisy_proposals(miss_rate: float, spurious: int, n_images: int, seed: int = 3):
    """Simulate a detector that misses real components and invents fake ones."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_images):
        pts = [
            ("pushpin", x + rng.normal(0, 0.006), y + rng.normal(0, 0.006))
            for (x, y) in GRID
            if rng.random() >= miss_rate
        ]
        pts += [("pushpin", rng.uniform(0, 1), rng.uniform(0, 1))
                for _ in range(rng.integers(0, spurious + 1))]
        frames.append(pts)
    images = [Image.new("RGB", (1000, 600)) for _ in frames]
    return images, frames


def test_consensus_recovers_slots_and_rejects_spurious_detections():
    from keyprompt.annotate.auto import consensus_annotations

    images, frames = _noisy_proposals(miss_rate=0.10, spurious=3, n_images=30)
    it = iter(frames)
    anns, report = consensus_annotations(images, lambda im: next(it), min_support=0.6)

    assert report.n_slots == len(GRID)
    assert report.dropped_clusters > 0  # the fake detections were filtered out
    assert len(anns) == len(images)


def test_bootstrapped_prior_matches_the_true_layout():
    from keyprompt.annotate.auto import consensus_annotations

    images, frames = _noisy_proposals(miss_rate=0.25, spurious=3, n_images=30)
    it = iter(frames)
    anns, _ = consensus_annotations(images, lambda im: next(it), min_support=0.6)
    g = NormalityGraph.build(anns, "pushpins")

    assert g.total_slots() == len(GRID)
    slots = g.classes["pushpin"].slots
    for truth in GRID:
        assert float(np.linalg.norm(slots - np.array(truth), axis=1).min()) < 0.02


def test_prior_anchors_on_the_most_complete_reference():
    """A sparse first annotation must not truncate the prior."""
    full = _refs(n=3)
    sparse = ImageAnnotation(
        "sparse", "pushpins", 1000, 600,
        [Keypoint("pushpin", x, y) for x, y in GRID[:5]],
    )
    g = NormalityGraph.build([sparse] + full, "pushpins")
    assert g.total_slots() == len(GRID)




# -- environment loading -------------------------------------------------


def test_dotenv_parses_quotes_exports_and_comments():
    from keyprompt.dotenv import parse_dotenv

    parsed = parse_dotenv(
        "# a comment\n"
        "\n"
        'export GEMINI_API_KEY="quoted-key"\n'
        "OPENROUTER_API_KEY=plain-key  # trailing comment\n"
        "GROQ_API_KEY='single'\n"
        "MALFORMED_LINE\n"
    )
    assert parsed["GEMINI_API_KEY"] == "quoted-key"
    assert parsed["OPENROUTER_API_KEY"] == "plain-key"
    assert parsed["GROQ_API_KEY"] == "single"
    assert "MALFORMED_LINE" not in parsed


def test_dotenv_does_not_override_the_real_environment(tmp_path, monkeypatch=None):
    """A value already exported must win over the file on disk."""
    import os

    from keyprompt.dotenv import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("KEYPROMPT_TEST_VAR=from-file\n")

    os.environ["KEYPROMPT_TEST_VAR"] = "from-shell"
    try:
        load_dotenv(env_file)
        assert os.environ["KEYPROMPT_TEST_VAR"] == "from-shell"
        load_dotenv(env_file, override=True)
        assert os.environ["KEYPROMPT_TEST_VAR"] == "from-file"
    finally:
        os.environ.pop("KEYPROMPT_TEST_VAR", None)

```

# Configuration files

## `configs/default.yaml`

_45 lines_

```yaml
# Default benchmark configuration.
# Every reported number should be traceable to one of these files plus a commit.

dataset_root: data/mvtec_loco_anomaly_detection
output_root: runs/gemini-flash-4shot
cache_root: .cache/vlm
categories:
  - pushpins
  - screw_bag
  - splicing_connectors
  - breakfast_box
  - juice_bottle
seed: 0
limit_per_split: null      # set to e.g. 5 for a smoke test
use_cache: true

provider:
  name: gemini             # gemini | openrouter | groq | echo
  model: gemini-2.0-flash
  api_key_env: GEMINI_API_KEY
  temperature: 0.0
  max_output_tokens: 2048
  max_retries: 4
  retry_base_delay: 4.0
  min_seconds_between_calls: 4.0   # free tier is roughly 15 requests/minute
  timeout: 120.0

image:
  max_side: 512            # ablated in the paper: 256 / 384 / 512 / 768
  keep_aspect: true
  jpeg_quality: 90

prior:
  n_shots: 4               # ablated: 1 / 2 / 4 / 8
  align: true
  min_sigma: 0.012

scoring:
  match_radius: 0.10
  w_missing: 1.0
  w_extra: 0.5
  w_displacement: 1.0
  w_edge: 0.8
  w_vlm: 0.6
  sigma_clip: 6.0

```

## `configs/smoke.yaml`

_16 lines_

```yaml
# Offline smoke test: no API key, no network, 3 images per subset.
# Verifies the plumbing end to end before spending quota.
dataset_root: data/mvtec_loco_anomaly_detection
output_root: runs/smoke
cache_root: .cache/vlm
categories: [pushpins]
seed: 0
limit_per_split: 3
use_cache: false
provider:
  name: echo
  model: offline-stub
  api_key_env: NONE
  min_seconds_between_calls: 0.0
image: {max_side: 256}
prior: {n_shots: 4}

```

## `configs/openrouter.yaml`

_12 lines_

```yaml
# Cross-model check on an open-weight VLM, to show the method is not tied to
# one proprietary backend.
dataset_root: data/mvtec_loco_anomaly_detection
output_root: runs/qwen25vl-4shot
categories: [pushpins, screw_bag, splicing_connectors, breakfast_box, juice_bottle]
provider:
  name: openrouter
  model: qwen/qwen2.5-vl-72b-instruct:free
  api_key_env: OPENROUTER_API_KEY
  min_seconds_between_calls: 6.0
image: {max_side: 512}
prior: {n_shots: 4}

```

## `configs/categories/pushpins.yaml`

_21 lines_

```yaml
name: pushpins
component_classes:
  - pushpin
normality_statement: |
  A correct part is a rectangular plastic tray divided into a regular grid of
  identical rectangular compartments. Exactly one pushpin sits inside every
  compartment. Pushpin colour and the exact angle at which a pin rests are not
  controlled and vary freely between correct parts.

  The part is defective when a compartment holds no pushpin, when a compartment
  holds more than one, or when a pin sits outside the compartment boundaries.
grouping_description: |
  Compartments are treated as slots arranged in fixed rows and columns. Each
  pushpin belongs to exactly one slot, and the slot layout is identical across
  every image because the tray is photographed in a fixed position. The spacing
  between neighbouring pins is therefore near-constant on a correct part.
ignored_variation:
  - variation in pushpin colour
  - the angle or orientation of an individual pin within its compartment
  - shadows, reflections and highlights on the plastic tray
  - dust, scratches or scuffs on the tray surface

```

## `configs/categories/screw_bag.yaml`

_23 lines_

```yaml
name: screw_bag
component_classes:
  - long_screw
  - short_screw
  - nut
  - washer
normality_statement: |
  A correct part is a transparent plastic bag containing exactly two long
  screws, two short screws, two hexagonal nuts and two washers. The items rest
  loosely, so their positions shift between images, but the bill of materials
  never changes.

  The part is defective when the count of any item type is wrong, when a screw
  of the wrong length is present, or when an item type is absent entirely.
grouping_description: |
  Because the contents are loose, position is only weakly informative here and
  the count of each component type carries most of the signal. Report every
  item you can see, with its class, and be careful to distinguish long screws
  from short ones by comparing shank length against the nuts beside them.
ignored_variation:
  - the position and rotation of items inside the bag
  - creases, folds and reflections in the plastic
  - the exact position of the printed label

```

## `configs/categories/splicing_connectors.yaml`

_23 lines_

```yaml
name: splicing_connectors
component_classes:
  - connector_terminal
  - cable_end
normality_statement: |
  A correct part is two identical splicing connector blocks joined by a single
  cable. Both blocks have the same number of terminals, the cable is connected
  to the mirrored terminal position on each block, and the cable colour
  corresponds to the block size according to a fixed convention.

  The part is defective when the two blocks have different terminal counts,
  when the cable is missing, when an extra cable is present, when the cable
  attaches at mismatched terminal positions, or when the cable colour does not
  match the block size.
grouping_description: |
  Treat each connector block as a group of terminal slots, and treat the cable
  as an edge linking one terminal on the left block to one on the right. The
  two endpoints of that edge must occupy mirrored positions within their
  respective blocks.
ignored_variation:
  - the exact curvature or slack of the cable between the two blocks
  - small rotations of the whole assembly on the background
  - dust and marks on the background surface

```

## `configs/categories/breakfast_box.yaml`

_25 lines_

```yaml
name: breakfast_box
component_classes:
  - orange
  - nectarine
  - cereal_region
  - banana_chip_region
  - almond_region
normality_statement: |
  A correct part is a rectangular box split into two compartments. The left
  compartment holds exactly two tangerines and one nectarine. The right
  compartment holds cereal, topped with banana chips and almonds in a roughly
  fixed ratio, covering the compartment evenly.

  The part is defective when fruit is missing, duplicated, or of the wrong
  type; when fruit appears in the wrong compartment; or when the proportion of
  cereal to banana chips and almonds departs clearly from the reference.
grouping_description: |
  The left compartment is a group of three fruit slots at fixed positions. The
  right compartment is treated as a single region: report the approximate
  centre of the cereal area, of the banana chip cluster, and of the almond
  cluster, rather than counting individual pieces.
ignored_variation:
  - the exact rotation of an individual piece of fruit
  - small differences in fruit size or surface colour
  - the fine-grained arrangement of individual cereal flakes

```

## `configs/categories/juice_bottle.yaml`

_23 lines_

```yaml
name: juice_bottle
component_classes:
  - fill_level
  - top_label
  - bottom_label
normality_statement: |
  A correct part is a glass bottle filled to a fixed level with juice, carrying
  one label near the top of the bottle and one near the bottom. The icon on the
  bottom label matches the juice type, and the top label carries the product
  banner.

  The part is defective when the fill level is too high or too low, when either
  label is missing, when the labels are swapped, when a label is damaged or
  misplaced, or when the icon does not match the juice colour.
grouping_description: |
  Report the fill level as a single point at the centre of the liquid surface,
  and each label as a point at its centre. The vertical distance between the
  fill point and the two label points is the quantity that identifies an
  incorrect fill.
ignored_variation:
  - reflections and highlights on the glass
  - small differences in background brightness
  - minor variation in juice shade

```

# Packaging

## `pyproject.toml`

_29 lines_

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "keyprompt-ad"
version = "0.1.0"
description = "Keypoint-grounded few-shot logical anomaly detection with vision-language models"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
  "numpy>=1.26", "scipy>=1.11", "Pillow>=10.0", "PyYAML>=6.0",
  "requests>=2.31", "matplotlib>=3.8",
]

[project.optional-dependencies]
gemini = ["google-genai>=0.3.0"]
dev = ["pytest>=8.0"]

[project.scripts]
keyprompt = "keyprompt.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

```

## `requirements.txt`

_13 lines_

```text
# core
numpy>=1.26
scipy>=1.11
Pillow>=10.0
PyYAML>=6.0
requests>=2.31
matplotlib>=3.8

# gemini backend (optional; only needed for provider.name = gemini)
google-genai>=0.3.0

# dev
pytest>=8.0

```

## `.env.example`

_7 lines_

```text
# Copy to .env and fill in. .env is gitignored: never commit a real key.
# If a key has ever been pasted into a chat window, an issue, or a notebook
# that was pushed, rotate it before using it here.

GEMINI_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=

```

## `.gitignore`

_24 lines_

```bash
# secrets
.env
*.key

# data and outputs
data/
runs/
.cache/
*.zip
*.tar.gz

# python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# os / editor
.DS_Store
.idea/
.vscode/

```

---

**Total: 4236 lines across 34 files.**
