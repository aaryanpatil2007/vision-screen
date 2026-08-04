import numpy as np
import pytest

from visionscreen.perception.eyes import eye_aspect_ratio, head_roll_deg, interocular_px

L = {"h": (33, 133), "v": [(159, 145), (158, 153)]}
R = {"h": (362, 263), "v": [(386, 374), (385, 380)]}


def synth_landmarks(open_h: float = 0.02, roll_rad: float = 0.0) -> np.ndarray:
    """478 landmarks with two synthetic eyes of controllable opening and roll.
    Roll rotates the whole face about the global center so measured roll == roll_rad."""
    lm = np.zeros((478, 3), np.float32)
    centers = {"L": np.array([0.35, 0.5]), "R": np.array([0.65, 0.5])}
    global_c = np.array([0.5, 0.5])
    rot = np.array(
        [[np.cos(roll_rad), -np.sin(roll_rad)], [np.sin(roll_rad), np.cos(roll_rad)]]
    )

    def put(idx, offset, center):
        p = center + np.array(offset, np.float32) - global_c
        lm[idx, :2] = global_c + rot @ p

    for key, spec in (("L", L), ("R", R)):
        c = centers[key]
        put(spec["h"][0], (-0.04, 0.0), c)
        put(spec["h"][1], (0.04, 0.0), c)
        for (top, bot), dx in zip(spec["v"], (-0.01, 0.01)):
            put(top, (dx, -open_h / 2), c)
            put(bot, (dx, open_h / 2), c)
    return lm


def test_ear_drops_when_eye_closes():
    open_ear = eye_aspect_ratio(synth_landmarks(open_h=0.02), "left")
    squint_ear = eye_aspect_ratio(synth_landmarks(open_h=0.006), "left")
    assert open_ear == pytest.approx(0.25, abs=0.02)
    assert squint_ear < open_ear / 2


def test_interocular_scales_with_frame_width():
    lm = synth_landmarks()
    # outer corners: 33 at x=0.31, 263 at x=0.69 → 0.38 of frame width
    assert interocular_px(lm, 640, 480) == pytest.approx(0.38 * 640, rel=0.05)


def test_head_roll_recovered():
    assert head_roll_deg(synth_landmarks()) == pytest.approx(0.0, abs=0.5)
    assert head_roll_deg(synth_landmarks(roll_rad=np.radians(10))) == pytest.approx(10.0, abs=1.0)
