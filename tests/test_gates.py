import numpy as np

from visionscreen.perception.landmarks import FaceFrame
from visionscreen.quality.gates import check_frame

from tests.test_eyes import synth_landmarks


def frame(brightness: int) -> np.ndarray:
    return np.full((480, 640, 3), brightness, dtype=np.uint8)


def test_no_face_fails():
    ff = FaceFrame(np.zeros((478, 3), np.float32), ok=False)
    r = check_frame(frame(128), ff)
    assert not r.passed
    assert "No face detected — face the camera." in r.failures


def test_good_frame_passes():
    ff = FaceFrame(synth_landmarks(), ok=True)  # interocular ≈ 243 px at 640 wide
    r = check_frame(frame(128), ff)
    assert r.passed and r.failures == []


def test_dark_frame_fails():
    ff = FaceFrame(synth_landmarks(), ok=True)
    r = check_frame(frame(10), ff)
    assert not r.passed
    assert "Lighting too dark — add light." in r.failures


def test_tiny_face_fails():
    lm = synth_landmarks()
    center = np.array([0.5, 0.5, 0.0], np.float32)
    shrunk = (lm - center) * 0.1 + center  # interocular ≈ 24 px
    r = check_frame(frame(128), FaceFrame(shrunk.astype(np.float32), ok=True))
    assert not r.passed
    assert "Move closer — your eyes are too small in frame." in r.failures
