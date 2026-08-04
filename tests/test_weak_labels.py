import numpy as np

from visionscreen.ml.model import IRIS_CLASS, PUPIL_CLASS, REFLEX_CLASS
from visionscreen.data.weak_labels import weak_label_eye_crop


def _synthetic_crop():
    """Dark pupil inside a mid-gray iris inside a bright sclera, plus a glint."""
    crop = np.full((60, 90), 210, np.uint8)
    yy, xx = np.mgrid[0:60, 0:90]
    r = np.hypot(xx - 45, yy - 30)
    crop[r <= 22] = 90     # iris
    crop[r <= 9] = 25      # pupil
    crop[(np.hypot(xx - 50, yy - 27)) <= 2.5] = 255  # corneal glint
    return crop


def test_labels_iris_disk_from_prior():
    crop = _synthetic_crop()
    mask = weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22)
    assert mask is not None
    labeled_iris = (mask == IRIS_CLASS).sum()
    assert labeled_iris > 200


def test_pupil_found_inside_iris():
    crop = _synthetic_crop()
    mask = weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22)
    ys, xs = np.nonzero(mask == PUPIL_CLASS)
    assert len(xs) > 20
    assert abs(xs.mean() - 45) < 4 and abs(ys.mean() - 30) < 4


def test_reflex_found_on_cornea():
    crop = _synthetic_crop()
    mask = weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22)
    ys, xs = np.nonzero(mask == REFLEX_CLASS)
    assert len(xs) >= 3
    assert np.hypot(xs.mean() - 50, ys.mean() - 27) < 4


def test_rejects_degenerate_input():
    assert weak_label_eye_crop(np.zeros((4, 4), np.uint8), (2, 2), 1.0) is None
    flat = np.full((60, 90), 128, np.uint8)
    # no intensity structure => no defensible pupil label => reject the sample
    assert weak_label_eye_crop(flat, (45, 30), 22) is None
