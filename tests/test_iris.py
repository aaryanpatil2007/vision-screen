import numpy as np
import pytest

from visionscreen.perception.iris import (
    detect_corneal_reflex,
    eye_crop,
    iris_center,
    iris_diameter_px,
)


def synth_iris_landmarks() -> np.ndarray:
    lm = np.zeros((478, 3), np.float32)
    # eye corners (match eyes.py specs)
    lm[33, :2], lm[133, :2] = (0.31, 0.5), (0.39, 0.5)
    lm[362, :2], lm[263, :2] = (0.61, 0.5), (0.69, 0.5)
    # iris centers + rings (left: 468 center, 469-472 ring; right: 473, 474-477)
    for center_idx, cx in ((468, 0.35), (473, 0.65)):
        lm[center_idx, :2] = (cx, 0.5)
        r = 0.02
        for i, (dx, dy) in enumerate([(r, 0), (0, r), (-r, 0), (0, -r)]):
            lm[center_idx + 1 + i, :2] = (cx + dx, 0.5 + dy)
    return lm


def test_iris_center_mapping():
    lm = synth_iris_landmarks()
    assert iris_center(lm, "left") == pytest.approx([0.35, 0.5])
    assert iris_center(lm, "right") == pytest.approx([0.65, 0.5])


def test_iris_diameter():
    lm = synth_iris_landmarks()
    assert iris_diameter_px(lm, "left", 1000, 1000) == pytest.approx(40.0, rel=0.05)


def test_eye_crop_contains_eye():
    lm = synth_iris_landmarks()
    frame = np.zeros((500, 1000, 3), np.uint8)
    crop, origin = eye_crop(frame, lm, "left")
    assert crop.size > 0
    ox, oy = origin
    assert ox <= 310 and ox + crop.shape[1] >= 390


def test_reflex_detected_at_bright_dot():
    crop = np.full((60, 80), 40, np.uint8)
    crop[30:33, 50:53] = 255
    xy = detect_corneal_reflex(crop)
    assert xy is not None
    assert xy[0] == pytest.approx(51, abs=1.5)
    assert xy[1] == pytest.approx(31, abs=1.5)


def test_no_reflex_in_dark_crop():
    crop = np.full((60, 80), 40, np.uint8)
    assert detect_corneal_reflex(crop) is None


def test_real_face_iris_inside_corners():
    pytest.importorskip("skimage")
    from skimage import data

    from visionscreen.perception.landmarks import LandmarkExtractor

    img = np.ascontiguousarray(data.astronaut()[:, :, ::-1])
    with LandmarkExtractor() as ex:
        ff = ex.extract(img)
    assert ff.ok
    for side, (a, b) in (("left", (33, 133)), ("right", (362, 263))):
        c = iris_center(ff.landmarks, side)
        lo, hi = sorted((ff.landmarks[a, 0], ff.landmarks[b, 0]))
        assert lo < c[0] < hi
