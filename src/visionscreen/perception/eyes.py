from __future__ import annotations

import numpy as np

_EYES = {
    "left": {"h": (33, 133), "v": [(159, 145), (158, 153)]},
    "right": {"h": (362, 263), "v": [(386, 374), (385, 380)]},
}
_OUTER_CORNERS = (33, 263)


def eye_aspect_ratio(landmarks: np.ndarray, side: str) -> float:
    spec = _EYES[side]
    h = np.linalg.norm(landmarks[spec["h"][0], :2] - landmarks[spec["h"][1], :2])
    if h == 0:
        return 0.0
    v = np.mean(
        [np.linalg.norm(landmarks[t, :2] - landmarks[b, :2]) for t, b in spec["v"]]
    )
    return float(v / h)


def interocular_px(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    a = landmarks[_OUTER_CORNERS[0], :2] * (frame_w, frame_h)
    b = landmarks[_OUTER_CORNERS[1], :2] * (frame_w, frame_h)
    return float(np.linalg.norm(a - b))


def head_roll_deg(landmarks: np.ndarray) -> float:
    a, b = landmarks[_OUTER_CORNERS[0], :2], landmarks[_OUTER_CORNERS[1], :2]
    dx, dy = b[0] - a[0], b[1] - a[1]
    return float(np.degrees(np.arctan2(dy, dx)))


# Upper-lid margin at the vertical meridian of the pupil. MediaPipe's eye
# contour puts 159 (left) and 386 (right) at the top of the palpebral aperture,
# which is where the lid margin crosses the line through the pupil centre —
# the point margin-reflex distance is defined from.
_UPPER_LID = {"left": 159, "right": 386}
_LOWER_LID = {"left": 145, "right": 374}


def upper_lid_point(landmarks: np.ndarray, side: str,
                    frame_w: int, frame_h: int) -> tuple[float, float]:
    p = landmarks[_UPPER_LID[side], :2] * (frame_w, frame_h)
    return float(p[0]), float(p[1])


def lower_lid_point(landmarks: np.ndarray, side: str,
                    frame_w: int, frame_h: int) -> tuple[float, float]:
    p = landmarks[_LOWER_LID[side], :2] * (frame_w, frame_h)
    return float(p[0]), float(p[1])


def palpebral_aperture_px(landmarks: np.ndarray, side: str,
                          frame_w: int, frame_h: int) -> float:
    """Vertical opening of the eye, in pixels.

    Used to reject frames captured mid-blink: a lid caught halfway down would
    otherwise read as ptosis, and blinks are frequent enough that a few of them
    in a session would produce a confident false finding.
    """
    up = upper_lid_point(landmarks, side, frame_w, frame_h)
    lo = lower_lid_point(landmarks, side, frame_w, frame_h)
    return abs(lo[1] - up[1])
