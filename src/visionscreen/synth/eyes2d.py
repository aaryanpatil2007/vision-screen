from __future__ import annotations

import cv2
import numpy as np

HVID_MM = 11.7  # horizontal visible iris diameter, population mean


def render_eye(
    width_px: int = 200,
    iris_diameter_px: float = 80.0,
    reflex_offset_mm: tuple[float, float] = (0.0, 0.0),
    pupil_ratio: float = 0.4,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Render one schematic eye: sclera, iris, pupil, specular corneal reflex.

    The reflex sits at pupil center + offset (mm converted via iris scale).
    Returns (BGR image, truth dict).
    """
    h = int(width_px * 0.6)
    img = np.full((h, width_px, 3), 235, np.uint8)  # sclera
    cx, cy = width_px / 2, h / 2
    px_per_mm = iris_diameter_px / HVID_MM

    cv2.circle(img, (int(cx), int(cy)), int(iris_diameter_px / 2), (90, 60, 30), -1)
    cv2.circle(img, (int(cx), int(cy)), int(iris_diameter_px / 2 * pupil_ratio), (10, 10, 10), -1)

    rx = cx + reflex_offset_mm[0] * px_per_mm
    ry = cy + reflex_offset_mm[1] * px_per_mm
    cv2.circle(img, (int(round(rx)), int(round(ry))), max(2, int(iris_diameter_px * 0.04)), (255, 255, 255), -1)

    if noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        noise = rng.normal(0, noise_sigma, img.shape)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    truth = {
        "pupil_center_px": (cx, cy),
        "iris_center_px": (cx, cy),
        "reflex_px": (rx, ry),
        "px_per_mm": px_per_mm,
        "iris_diameter_px": iris_diameter_px,
    }
    return img, truth


def render_eye_pair(
    offset_left_mm: tuple[float, float] = (0.0, 0.0),
    offset_right_mm: tuple[float, float] = (0.0, 0.0),
    width_px: int = 200,
    iris_diameter_px: float = 80.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    left, tl = render_eye(width_px, iris_diameter_px, offset_left_mm, noise_sigma=noise_sigma, rng=rng)
    right, tr = render_eye(width_px, iris_diameter_px, offset_right_mm, noise_sigma=noise_sigma, rng=rng)
    img = np.hstack([left, right])
    tr = dict(tr)
    tr["pupil_center_px"] = (tr["pupil_center_px"][0] + width_px, tr["pupil_center_px"][1])
    tr["iris_center_px"] = tr["pupil_center_px"]
    tr["reflex_px"] = (tr["reflex_px"][0] + width_px, tr["reflex_px"][1])
    return img, {"left": tl, "right": tr}
