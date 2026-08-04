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
# Physiological bounds, expressed as pupil AREA over iris-disk area. A pupil
# is 2-8 mm across an ~11.7 mm iris, i.e. a diameter ratio of 0.17-0.68 and
# therefore an area ratio of 0.029-0.46. The previous 0.75 area bound allowed a
# 0.87 diameter ratio — near-total dilation — and measurement showed the labels
# sitting at 0.81 diameter ratio (0.66 area): Otsu's dark mode inside the iris
# disk absorbs dark iris pigment along with the pupil.
MIN_PUPIL_FRACTION = 0.029
MAX_PUPIL_FRACTION = 0.46
# Typical dim-adapted pupil is ~0.45 of iris diameter, i.e. 0.20 of its area.
# Selecting the first threshold merely *under* the ceiling clamps every label
# to the boundary (measured: 0.645 +- 0.014 diameter ratio, an artefact of the
# bound rather than a measurement), so choose the candidate whose area lands
# nearest this target instead.
TARGET_PUPIL_AREA_FRACTION = 0.20
REFLEX_PERCENTILE = 99.0
MIN_PUPIL_IRIS_CONTRAST = 12.0   # pupil must be darker than the iris annulus
MAX_SATURATED_FRACTION = 0.35    # blown-out crop (glasses glare) -> unusable


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

    # Spectacle glare blows out the eye region entirely; anything labeled on a
    # saturated crop is a reflection, not anatomy.
    if float((disk_vals >= 250).mean()) > MAX_SATURATED_FRACTION:
        return None

    # --- pupil: the darkest core of physiological size within the iris disk ---
    # Otsu alone finds the dark *mode*, which in a pigmented iris is pupil plus
    # surrounding stroma. Instead, walk the threshold down from Otsu until the
    # largest dark component falls inside the physiological area band; that is
    # the intensity level at which the core is pupil-sized rather than
    # pupil-plus-iris.
    disk_area = int(iris_disk.sum())
    vals = disk_vals.astype(np.uint8)
    otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pupil = None
    idx = -1
    labels = stats = cents = None
    best_gap = float("inf")
    candidates = [otsu] + [
        float(np.percentile(disk_vals, q))
        for q in (45, 40, 35, 30, 26, 22, 18, 15, 12, 9, 7, 5, 3)
    ]
    for thr in candidates:
        dark = iris_disk & (crop_gray <= thr)
        n, lab, st, ct = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        frac = st[i, cv2.CC_STAT_AREA] / disk_area
        if not (MIN_PUPIL_FRACTION <= frac <= MAX_PUPIL_FRACTION):
            continue
        gap = abs(frac - TARGET_PUPIL_AREA_FRACTION)
        if gap < best_gap:
            best_gap = gap
            pupil, idx, labels, stats, cents = lab == i, i, lab, st, ct
    if pupil is None:
        return None

    # the pupil is concentric with the iris; a far-off blob is an eyelash/shadow
    py, px = cents[idx][1], cents[idx][0]
    if np.hypot(px - iris_center[0], py - iris_center[1]) > iris_radius * 0.6:
        return None

    # A pupil is necessarily darker than the iris around it. Without this, a
    # bright spectacle reflection inside the disk can win the largest-component
    # vote and get labeled as pupil (observed on real eyeglass wearers).
    annulus = iris_disk & ~pupil
    if annulus.sum() < 20:
        return None
    pupil_med = float(np.median(crop_gray[pupil]))
    iris_med = float(np.median(crop_gray[annulus]))
    if iris_med - pupil_med < MIN_PUPIL_IRIS_CONTRAST:
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
