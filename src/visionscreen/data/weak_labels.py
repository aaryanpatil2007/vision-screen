"""Weak supervision: turn real eye photos into segmentation labels.

Real datasets ship gaze targets, not pupil masks. But two priors are strong
enough to label them without a human:

1. **Geometric** — MediaPipe's iris landmarks are a validated circle fit, so
   the iris disk is known a priori.
2. **Photometric** — *within* that disk the pupil is the dark mode and the
   corneal reflex is the bright mode. Restricting Otsu to the disk removes
   the confounders (skin, sclera, stray speculars) that make global
   thresholding fail on real webcam frames.

Samples where the priors disagree are rejected rather than labeled badly:
a wrong label is worse than a missing one.
"""
from __future__ import annotations

import cv2
import numpy as np

from visionscreen.ml.model import IRIS_CLASS, PUPIL_CLASS, REFLEX_CLASS

OCULAR_CLASS = 1
MIN_CROP_PX = 20
MIN_PUPIL_FRACTION = 0.02   # of iris disk area
MAX_PUPIL_FRACTION = 0.75
REFLEX_PERCENTILE = 99.0


def weak_label_eye_crop(
    crop_gray: np.ndarray,
    iris_center: tuple[float, float],
    iris_radius: float,
) -> np.ndarray | None:
    h, w = crop_gray.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX or iris_radius < 4:
        return None

    ys, xs = np.mgrid[0:h, 0:w]
    r = np.hypot(xs - iris_center[0], ys - iris_center[1])
    iris_disk = r <= iris_radius
    if iris_disk.sum() < 50:
        return None

    mask = np.zeros((h, w), np.uint8)
    # everything inside a generous ocular ellipse counts as ocular surface
    ocular = r <= iris_radius * 2.6
    mask[ocular] = OCULAR_CLASS
    mask[iris_disk] = IRIS_CLASS

    disk_vals = crop_gray[iris_disk]
    spread = float(disk_vals.max()) - float(disk_vals.min())
    if spread < 25:
        return None  # no photometric structure — cannot defend a pupil label

    # --- pupil: Otsu restricted to the iris disk, keep the dark side ---
    vals = disk_vals.astype(np.uint8)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = iris_disk & (crop_gray <= thr)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[idx, cv2.CC_STAT_AREA]
    frac = area / iris_disk.sum()
    if not (MIN_PUPIL_FRACTION <= frac <= MAX_PUPIL_FRACTION):
        return None
    pupil = labels == idx

    # the pupil is concentric with the iris; a far-off blob is an eyelash/shadow
    py, px = cents[idx][1], cents[idx][0]
    if np.hypot(px - iris_center[0], py - iris_center[1]) > iris_radius * 0.6:
        return None
    mask[pupil] = PUPIL_CLASS

    # --- corneal reflex: bright mode inside the disk, excluding pupil edge halo ---
    bright_thr = np.percentile(disk_vals, REFLEX_PERCENTILE)
    if bright_thr >= 180:
        bright = iris_disk & (crop_gray >= max(bright_thr, 200))
        nb, blabels, bstats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
        if nb >= 2:
            bidx = 1 + int(np.argmax(bstats[1:, cv2.CC_STAT_AREA]))
            if 2 <= bstats[bidx, cv2.CC_STAT_AREA] <= iris_disk.sum() * 0.25:
                mask[blabels == bidx] = REFLEX_CLASS
    return mask
