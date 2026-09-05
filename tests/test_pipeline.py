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
