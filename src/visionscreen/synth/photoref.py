from __future__ import annotations

import numpy as np

# Eccentric photorefraction (Bobier & Braddick): for defocus A (diopters,
# relative to the camera plane), flash eccentricity e (m), camera distance d (m)
# and pupil radius r (m), the bright reflex extent from the flash-side pupil
# edge is  w = 2r − e / (d·|A|), clipped to [0, 2r]. w == 0 is the dead zone.
BRIGHT = 220
BASE = 60
BRIGHT_THRESHOLD = 140  # measurement threshold between BASE and BRIGHT


def crescent_width_px(
    A_diopters: float,
    pupil_radius_px: float,
    e_m: float = 0.005,
    d_m: float = 0.5,
    px_per_m: float = 8000.0,
) -> float:
    if abs(A_diopters) < 1e-9:
        return 0.0
    r_m = pupil_radius_px / px_per_m
    w_m = 2.0 * r_m - e_m / (d_m * abs(A_diopters))
    return float(np.clip(w_m, 0.0, 2.0 * r_m) * px_per_m)


def meridional_defocus(S: float, C: float, axis_deg: float, theta_deg: float) -> float:
    return S + C * np.sin(np.radians(theta_deg - axis_deg)) ** 2


def render_reflex(
    pupil_radius_px: float,
    S: float,
    C: float = 0.0,
    axis_deg: float = 0.0,
    e_m: float = 0.005,
    d_m: float = 0.5,
    px_per_m: float = 8000.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Render a pupil red-reflex image with an eccentric-photorefraction crescent.

    Flash is offset along +x. Crescent renders on the +x side for myopic
    defocus (S < 0), −x side for hyperopic (S > 0). Per-pixel meridian angle
    modulates the width via the meridional defocus (astigmatism morphology —
    a documented simplification of the full double-pass optics).
    """
    r = float(pupil_radius_px)
    size = int(np.ceil(4 * r))
    img = np.zeros((size, size), np.float32)
    cy = cx = size / 2.0
    side = 1.0 if S < 0 else -1.0

    ys, xs = np.mgrid[0:size, 0:size]
    dx, dy = xs - cx, ys - cy
    rho = np.hypot(dx, dy)
    in_pupil = rho <= r
    img[in_pupil] = BASE

    theta = np.degrees(np.arctan2(dy, dx))  # pixel meridian
    A_theta = meridional_defocus(S, C, axis_deg, theta)
    w_theta = np.zeros_like(A_theta)
    nz = np.abs(A_theta) > 1e-9
    r_m = r / px_per_m
    w_m = 2.0 * r_m - e_m / (d_m * np.abs(A_theta[nz]))
    w_theta[nz] = np.clip(w_m, 0.0, 2.0 * r_m) * px_per_m

    bright = in_pupil & (w_theta > 0.5) & (side * dx >= r - w_theta)
    img[bright] = BRIGHT

    if noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    truth = {
        "center_px": (cx, cy),
        "pupil_radius_px": r,
        "S": S,
        "C": C,
        "axis_deg": axis_deg,
        "side": side,
        "e_m": e_m,
        "d_m": d_m,
        "px_per_m": px_per_m,
    }
    return img, truth
