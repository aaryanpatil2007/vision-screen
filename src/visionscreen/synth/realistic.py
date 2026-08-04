"""Domain-randomized synthetic eye-crop generator with segmentation labels.

The schematic renderers in eyes2d/photoref exist to validate the *physics*;
this one exists to TRAIN perception. It therefore models the nuisance
variation that breaks threshold-based detection on real webcams: eyelids and
lashes, skin tone, iris texture and color, off-axis gaze, motion blur, JPEG-ish
noise, uneven illumination, and stray specular highlights that are NOT the
corneal reflex (the exact failure mode that killed alignment on real video).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionscreen.ml.model import IRIS_CLASS, PUPIL_CLASS, REFLEX_CLASS

OCULAR_CLASS = 1


@dataclass(frozen=True)
class EyeParams:
    size: tuple[int, int]          # (h, w)
    iris_radius: float
    pupil_ratio: float             # pupil radius / iris radius
    iris_center: tuple[float, float]
    iris_gray: float
    sclera_gray: float
    skin_gray: float
    lid_upper: float               # fraction of height covered by upper lid
    lid_lower: float
    reflex_intensity: float        # 0 = absent
    reflex_offset: tuple[float, float]   # in iris radii
    reflex_radius: float
    distractor_specular: int       # count of non-corneal bright spots
    blur_sigma: float
    noise_sigma: float
    illum_gradient: float          # -1..1 left-right brightness ramp
    exposure: float                # multiplicative


def sample_params(rng: np.random.Generator) -> EyeParams:
    w = int(rng.integers(48, 160))
    h = int(w * rng.uniform(0.55, 0.85))
    iris_r = w * rng.uniform(0.16, 0.30)
    exposure = float(rng.uniform(0.35, 1.45))
    return EyeParams(
        size=(h, w),
        iris_radius=iris_r,
        pupil_ratio=float(rng.uniform(0.28, 0.75)),
        iris_center=(
            float(w * rng.uniform(0.35, 0.65)),
            float(h * rng.uniform(0.40, 0.62)),
        ),
        iris_gray=float(rng.uniform(35, 120)),
        sclera_gray=float(rng.uniform(150, 235)),
        skin_gray=float(rng.uniform(60, 200)),
        lid_upper=float(rng.uniform(0.03, 0.34)),
        lid_lower=float(rng.uniform(0.03, 0.28)),
        reflex_intensity=float(rng.uniform(0.0, 1.0)),
        reflex_offset=(float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.5, 0.5))),
        reflex_radius=float(max(1.0, iris_r * rng.uniform(0.05, 0.18))),
        distractor_specular=int(rng.integers(0, 3)),
        blur_sigma=float(rng.uniform(0.0, 1.8)),
        noise_sigma=float(rng.uniform(1.0, 14.0)),
        illum_gradient=float(rng.uniform(-0.45, 0.45)),
        exposure=exposure,
    )


def _iris_texture(radius: float, rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Radial crypt-like streaks — gives the net real texture to key on."""
    h, w = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(ys - h / 2, xs - w / 2)
    freq = rng.uniform(14, 34)
    phase = rng.uniform(0, 2 * np.pi)
    return np.sin(ang * freq + phase) * rng.uniform(3, 11)


def render_labeled_eye(
    p: EyeParams, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, EyeParams]:
    rng = rng or np.random.default_rng(0)
    h, w = p.size
    img = np.full((h, w), p.skin_gray, np.float32)
    mask = np.zeros((h, w), np.uint8)

    cx, cy = p.iris_center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # --- ocular surface (almond aperture between the lids) ---
    ap_h = h * (1.0 - p.lid_upper - p.lid_lower)
    ap_cy = h * p.lid_upper + ap_h / 2
    aperture = ((xs - w / 2) / (w * 0.48)) ** 2 + ((ys - ap_cy) / max(ap_h / 2, 1)) ** 2 <= 1.0
    img[aperture] = p.sclera_gray
    mask[aperture] = OCULAR_CLASS

    # --- iris (clipped by the aperture) ---
    r = np.hypot(xs - cx, ys - cy)
    iris = (r <= p.iris_radius) & aperture
    tex = _iris_texture(p.iris_radius, rng, (h, w))
    img[iris] = np.clip(p.iris_gray + tex[iris], 0, 255)
    # limbus darkening — a real cue the classical detector never used
    limbus = iris & (r > p.iris_radius * 0.85)
    img[limbus] *= 0.75
    mask[iris] = IRIS_CLASS

    # --- pupil ---
    pupil_r = p.iris_radius * p.pupil_ratio
    pupil = (r <= pupil_r) & aperture
    img[pupil] = np.clip(rng.uniform(8, 34), 0, 255)
    mask[pupil] = PUPIL_CLASS

    # --- corneal reflex (on the cornea, i.e. within the iris) ---
    if p.reflex_intensity > 0.15:
        rx = cx + p.reflex_offset[0] * p.iris_radius
        ry = cy + p.reflex_offset[1] * p.iris_radius
        refl = (np.hypot(xs - rx, ys - ry) <= p.reflex_radius) & aperture
        if refl.sum() >= 3:
            img[refl] = np.clip(190 + 65 * p.reflex_intensity, 0, 255)
            mask[refl] = REFLEX_CLASS

    # --- distractor speculars: bright spots on skin/sclera, NOT the reflex ---
    for _ in range(p.distractor_specular):
        dx = rng.uniform(0, w)
        dy = rng.uniform(0, h)
        if np.hypot(dx - cx, dy - cy) < p.iris_radius * 1.3:
            continue  # keep them off the cornea so the label stays truthful
        rad = rng.uniform(1.0, max(1.5, p.iris_radius * 0.35))
        spot = np.hypot(xs - dx, ys - dy) <= rad
        img[spot] = np.clip(rng.uniform(200, 255), 0, 255)

    # --- eyelashes along the upper lid ---
    lash_y = int(h * p.lid_upper)
    for _ in range(int(rng.integers(0, 14))):
        x0 = int(rng.uniform(0, w))
        length = int(rng.uniform(2, max(3, h * 0.18)))
        cv2.line(img, (x0, lash_y), (x0 + int(rng.uniform(-3, 3)), lash_y + length),
                 float(rng.uniform(10, 60)), 1)

    # --- photometric nuisances ---
    ramp = 1.0 + p.illum_gradient * (xs / max(w - 1, 1) - 0.5) * 2.0
    img = img * ramp * p.exposure
    if p.blur_sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), p.blur_sigma)
    img = img + rng.normal(0, p.noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask, p


def generate_dataset(n: int, seed: int = 0):
    """Yield (image, mask) pairs."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        img, mask, _ = render_labeled_eye(sample_params(rng), rng)
        yield img, mask
