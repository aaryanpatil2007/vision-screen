import numpy as np
import pytest

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


def test_rejects_glare_labeled_as_pupil():
    """A bright spectacle reflection must never be labeled as pupil."""
    crop = np.full((60, 90), 120, np.uint8)
    yy, xx = np.mgrid[0:60, 0:90]
    r = np.hypot(xx - 45, yy - 30)
    crop[r <= 22] = 130                       # iris, no dark pupil at all
    crop[np.hypot(xx - 45, yy - 30) <= 8] = 255   # big central glare blob
    assert weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22) is None


def test_rejects_blown_out_crop():
    crop = np.full((60, 90), 253, np.uint8)
    crop[28:32, 43:47] = 20
    assert weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22) is None


def test_still_accepts_valid_dark_pupil():
    crop = _synthetic_crop()
    assert weak_label_eye_crop(crop, iris_center=(45, 30), iris_radius=22) is not None


def _ratio(mask):
    import numpy as np
    npx = (mask == PUPIL_CLASS).sum()
    nir = ((mask == IRIS_CLASS) | (mask == PUPIL_CLASS) | (mask == REFLEX_CLASS)).sum()
    return float(np.sqrt(npx / nir)) if nir else 0.0


def test_pupil_label_is_physiologically_sized():
    """Otsu's dark mode inside a pigmented iris absorbs stroma along with the
    pupil: measured labels sat at 0.81 of iris diameter, where a dim-adapted
    pupil is 0.35-0.55. The threshold search now targets a physiological area."""
    crop = np.full((80, 110), 200, np.uint8)
    yy, xx = np.mgrid[0:80, 0:110]
    r = np.hypot(xx - 55, yy - 40)
    crop[r <= 30] = 95      # iris
    crop[r <= 22] = 70      # dark iris stroma — must NOT be labelled pupil
    crop[r <= 13] = 20      # true pupil: 13/30 = 0.43 of iris radius
    mask = weak_label_eye_crop(crop, iris_center=(55, 40), iris_radius=30)
    assert mask is not None
    assert 0.30 <= _ratio(mask) <= 0.60, _ratio(mask)


def test_bounds_bracket_real_pupil_physiology():
    from visionscreen.data.weak_labels import (
        MAX_PUPIL_FRACTION, MIN_PUPIL_FRACTION, TARGET_PUPIL_AREA_FRACTION,
    )
    # a 2-8 mm pupil on an 11.7 mm iris is 0.17-0.68 of diameter
    assert MIN_PUPIL_FRACTION == pytest.approx((0.17) ** 2, abs=0.005)
    assert MAX_PUPIL_FRACTION == pytest.approx((0.68) ** 2, abs=0.005)
    assert MIN_PUPIL_FRACTION < TARGET_PUPIL_AREA_FRACTION < MAX_PUPIL_FRACTION
