"""Composite a controlled corneal glint onto a REAL eye crop.

The gap between synthetic benchmarks and real photographs is appearance: skin
texture, iris pigment, sensor noise, blur, ambient reflections. The gap between
real photographs and a usable measurement is ground truth: nobody labels the
prism dioptres in a Commons photo.

This closes both at once. Take a real eye crop, place a specular highlight at a
*known* decentration from the measured iris centre, and the pipeline can be
scored on real appearance against an exact answer. It is the same trick a
clinical transilluminator plays — supply your own light so its position is
known — done in software.

What it does not simulate: the optics of a genuinely deviated eye. A real
strabismic eye is rotated, so its iris and lid geometry differ too. This tests
the *measurement chain* (find the iris, find the glint, convert to dioptres) on
real images, not the clinical inference from a truly deviated eye.
"""
from __future__ import annotations

import cv2
import numpy as np

from visionscreen.synth.eyes2d import HVID_MM


def add_glint(
    crop_gray: np.ndarray,
    iris_center: tuple[float, float],
    iris_diameter_px: float,
    decentration_mm: tuple[float, float],
    radius_px: float | None = None,
    intensity: int = 255,
    softness: float = 0.6,
) -> np.ndarray:
    """Place a specular highlight at a known offset from the iris centre.

    Returns a copy; the original is untouched.
    """
    out = crop_gray.astype(np.float32).copy()
    h, w = out.shape[:2]
    px_per_mm = iris_diameter_px / HVID_MM
    if px_per_mm <= 0:
        return crop_gray.copy()

    gx = iris_center[0] + decentration_mm[0] * px_per_mm
    gy = iris_center[1] + decentration_mm[1] * px_per_mm
    r = radius_px if radius_px is not None else max(1.5, iris_diameter_px * 0.06)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xs - gx, ys - gy)
    # a real catchlight has a bright core and a soft edge, not a hard disk
    core = np.clip((r - d) / max(r * softness, 1e-3), 0.0, 1.0)
    out = np.maximum(out, core * intensity)
    return np.clip(out, 0, 255).astype(np.uint8)


def suppress_existing_speculars(
    crop_gray: np.ndarray, iris_center: tuple[float, float],
    iris_diameter_px: float, percentile: float = 97.0,
) -> np.ndarray:
    """Dim pre-existing highlights inside the iris so the injected glint is the
    only corneal reflection — otherwise ground truth is ambiguous."""
    out = crop_gray.astype(np.float32).copy()
    h, w = out.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    disk = np.hypot(xs - iris_center[0], ys - iris_center[1]) <= iris_diameter_px / 2
    if not disk.any():
        return crop_gray.copy()
    vals = out[disk]
    thr = float(np.percentile(vals, percentile))
    median = float(np.median(vals))
    hot = disk & (out >= thr)
    if hot.any():
        out[hot] = median
        out = cv2.GaussianBlur(out, (0, 0), 1.0) * hot + out * (~hot)
    return np.clip(out, 0, 255).astype(np.uint8)
