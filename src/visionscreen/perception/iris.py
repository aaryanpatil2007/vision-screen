from __future__ import annotations

import numpy as np

# Verified empirically (see Plan 2): iris center 468 belongs to the eye with
# corners 33/133 ("left" in eyes.py), 473 to corners 362/263 ("right").
_IRIS = {
    "left": {"center": 468, "ring": (469, 470, 471, 472), "corners": (33, 133)},
    "right": {"center": 473, "ring": (474, 475, 476, 477), "corners": (362, 263)},
}

REFLEX_MIN_INTENSITY = 200
REFLEX_PERCENTILE = 96.0


def iris_center(landmarks: np.ndarray, side: str) -> np.ndarray:
    return landmarks[_IRIS[side]["center"], :2].copy()


def iris_diameter_px(landmarks: np.ndarray, side: str, frame_w: int, frame_h: int) -> float:
    ring = landmarks[list(_IRIS[side]["ring"]), :2] * (frame_w, frame_h)
    center = landmarks[_IRIS[side]["center"], :2] * (frame_w, frame_h)
    return float(2.0 * np.linalg.norm(ring - center, axis=1).mean())


def eye_crop(
    frame_bgr: np.ndarray, landmarks: np.ndarray, side: str, pad: float = 0.5
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame_bgr.shape[:2]
    a, b = _IRIS[side]["corners"]
    pa, pb = landmarks[a, :2] * (w, h), landmarks[b, :2] * (w, h)
    width = abs(pb[0] - pa[0])
    cx, cy = (pa + pb) / 2
    half_w = width * (0.5 + pad)
    half_h = width * 0.5
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, y1 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    return frame_bgr[y0:y1, x0:x1], (x0, y0)


def detect_corneal_reflex(crop_gray: np.ndarray) -> tuple[float, float] | None:
    if crop_gray.size == 0:
        return None
    thresh = max(float(np.percentile(crop_gray, REFLEX_PERCENTILE)), REFLEX_MIN_INTENSITY)
    ys, xs = np.nonzero(crop_gray >= thresh)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
