import numpy as np
import pytest

from visionscreen.synth.eyes2d import render_eye, render_eye_pair


def test_reflex_at_truth_position():
    img, truth = render_eye(width_px=200, iris_diameter_px=80, reflex_offset_mm=(0.5, 0.0))
    x, y = truth["reflex_px"]
    # a bright pixel exists at the truth location
    gray = img.mean(axis=2)
    assert gray[int(round(y)), int(round(x))] > 200


def test_px_per_mm_scale():
    _, truth = render_eye(width_px=200, iris_diameter_px=80, reflex_offset_mm=(1.0, 0.0))
    assert truth["px_per_mm"] == pytest.approx(80 / 11.7, rel=1e-6)
    dx = truth["reflex_px"][0] - truth["pupil_center_px"][0]
    assert dx == pytest.approx(80 / 11.7, abs=0.6)


def test_pair_offsets_independent():
    img, truth = render_eye_pair(offset_left_mm=(0.0, 0.0), offset_right_mm=(1.5, 0.0))
    dl = np.subtract(truth["left"]["reflex_px"], truth["left"]["pupil_center_px"])
    dr = np.subtract(truth["right"]["reflex_px"], truth["right"]["pupil_center_px"])
    assert np.linalg.norm(dl) < 1.0
    assert np.linalg.norm(dr) == pytest.approx(1.5 * truth["right"]["px_per_mm"], abs=1.0)
    assert img.shape[1] >= 2 * img.shape[0]
