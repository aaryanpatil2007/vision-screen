import numpy as np

from visionscreen.ml.model import IRIS_CLASS, PUPIL_CLASS, REFLEX_CLASS
from visionscreen.synth.realistic import EyeParams, render_labeled_eye, sample_params


def test_returns_image_and_mask_of_same_size():
    img, mask, params = render_labeled_eye(sample_params(np.random.default_rng(0)))
    assert img.shape == mask.shape
    assert img.dtype == np.uint8
    assert mask.dtype == np.uint8


def test_mask_has_pupil_and_iris():
    img, mask, p = render_labeled_eye(sample_params(np.random.default_rng(1)))
    assert (mask == PUPIL_CLASS).sum() > 20
    assert (mask == IRIS_CLASS).sum() > (mask == PUPIL_CLASS).sum()


def test_pupil_inside_iris_geometrically():
    rng = np.random.default_rng(2)
    img, mask, p = render_labeled_eye(sample_params(rng))
    ys, xs = np.nonzero(mask == PUPIL_CLASS)
    py, px = ys.mean(), xs.mean()
    iy, ix = np.nonzero(mask == IRIS_CLASS)
    # pupil centroid sits within the iris bounding box
    assert ix.min() <= px <= ix.max()
    assert iy.min() <= py <= iy.max()


def test_reflex_present_when_requested():
    p = sample_params(np.random.default_rng(3))
    p = EyeParams(**{**p.__dict__, "reflex_intensity": 1.0})
    img, mask, _ = render_labeled_eye(p)
    assert (mask == REFLEX_CLASS).sum() >= 4


def test_variation_across_seeds():
    a, _, _ = render_labeled_eye(sample_params(np.random.default_rng(10)))
    b, _, _ = render_labeled_eye(sample_params(np.random.default_rng(11)))
    assert a.shape != b.shape or np.abs(a.astype(int) - b.astype(int)).mean() > 5


def test_dark_and_bright_lighting_both_generated():
    means = [
        render_labeled_eye(sample_params(np.random.default_rng(s)))[0].mean()
        for s in range(40)
    ]
    assert min(means) < 90 and max(means) > 130  # wide illumination coverage
