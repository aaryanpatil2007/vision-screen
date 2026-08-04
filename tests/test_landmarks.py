import numpy as np
import pytest

from visionscreen.perception.landmarks import FaceFrame, LandmarkExtractor


def test_no_face_returns_not_ok():
    with LandmarkExtractor() as ex:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ff = ex.extract(blank)
    assert isinstance(ff, FaceFrame)
    assert ff.ok is False
    assert ff.landmarks.shape == (478, 3)


def test_real_face_detected():
    pytest.importorskip("skimage")
    from skimage import data

    face_rgb = data.astronaut()  # public-domain photo containing a real face
    face_bgr = face_rgb[:, :, ::-1].copy()
    with LandmarkExtractor() as ex:
        ff = ex.extract(face_bgr)
    assert ff.ok is True
    # normalized coords in [0, 1] for a centered face
    assert 0.0 < ff.landmarks[:, 0].mean() < 1.0
    assert 0.0 < ff.landmarks[:, 1].mean() < 1.0
