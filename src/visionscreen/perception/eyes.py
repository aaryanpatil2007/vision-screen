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
