from __future__ import annotations

import statistics

import numpy as np

from visionscreen.report import Finding
from visionscreen.synth.photoref import BRIGHT_THRESHOLD

MAX_ABS_DIOPTERS = 10.0
PROFILE_ANGLES_DEG = np.arange(-60.0, 60.1, 5.0)
MIN_PROFILE_POINTS = 5


def _ray_depth(img: np.ndarray, cx: float, cy: float, radius: float,
               u: float, theta_deg: float) -> float:
    """Depth (from pupil edge inward) of the contiguous bright run along the ray
    at meridian theta on the crescent side (u = ±1 along +x flash axis)."""
    t = np.radians(theta_deg)
    depth = 0.0
    # full chord: wide crescents extend past the pupil center to the far side
    for rho in np.arange(radius, -radius, -0.5):
        x = int(round(cx + u * rho * np.cos(t)))
        y = int(round(cy + rho * np.sin(t)))
        if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
            continue
        if img[y, x] > BRIGHT_THRESHOLD:
            depth = radius - rho + 0.5
        else:
            if depth > 0:
                break
    return depth


def _depth_to_width(depth: float, radius: float, theta_deg: float) -> float:
    c = np.cos(np.radians(theta_deg))
    return float(depth * c + radius * (1.0 - c))


def invert_width(w_px: float, pupil_radius_px: float, e_m: float, d_m: float,
                 px_per_m: float) -> float | None:
    if w_px <= 0.5:
        return None
    w_m = w_px / px_per_m
    r_m = pupil_radius_px / px_per_m
    denom = 2.0 * r_m - w_m
    if denom <= 1e-9:
        return MAX_ABS_DIOPTERS
    return float(min(e_m / (d_m * denom), MAX_ABS_DIOPTERS))


def fit_srx(profile: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Fit |A|(θ) = S_abs + C·sin²(θ−axis) via the cos2θ/sin2θ linearization.
    Returns (S_abs, C, axis_deg in [0, 180))."""
    thetas = np.radians([t for t, _ in profile])
    values = np.array([v for _, v in profile])
    X = np.column_stack([np.ones_like(thetas), np.cos(2 * thetas), np.sin(2 * thetas)])
    c0, c1, c2 = np.linalg.lstsq(X, values, rcond=None)[0]
    half_c = float(np.hypot(c1, c2))
    S_abs = float(c0 - half_c)
    C = float(2.0 * half_c)
    axis = float(np.degrees(0.5 * np.arctan2(-c2, -c1)) % 180.0)
    return S_abs, C, axis


def measure_reflex(
    img_gray: np.ndarray,
    center_px: tuple[float, float],
    pupil_radius_px: float,
    e_m: float = 0.005,
    d_m: float = 0.5,
    px_per_m: float = 8000.0,
) -> tuple[float, float, float] | None:
    """Measure (S_signed, C, axis_deg) from one pupil reflex image.
    Returns None when no crescent is visible (dead zone or no signal)."""
    cx, cy = center_px
    ys, xs = np.mgrid[0 : img_gray.shape[0], 0 : img_gray.shape[1]]
    in_pupil = (xs - cx) ** 2 + (ys - cy) ** 2 <= pupil_radius_px**2
    bright = in_pupil & (img_gray > BRIGHT_THRESHOLD)
    if bright.sum() < 5:
        return None
    # crescent side: sign of the bright centroid's x-offset from pupil center
    u = 1.0 if (xs[bright].mean() - cx) >= 0 else -1.0

    profile: list[tuple[float, float]] = []
    for theta in PROFILE_ANGLES_DEG:
        depth = _ray_depth(img_gray, cx, cy, pupil_radius_px, u, theta)
        w = _depth_to_width(depth, pupil_radius_px, theta)
        a = invert_width(w, pupil_radius_px, e_m, d_m, px_per_m)
        if a is not None:
            profile.append((theta, a))
    if len(profile) < MIN_PROFILE_POINTS:
        return None

    S_abs, C, axis = fit_srx(profile)
    # myopic defocus (S < 0) puts the crescent on the flash side (+x)
    sign = -1.0 if u > 0 else 1.0
    return sign * S_abs, C, axis


def score_photoref(
    estimates: list[tuple[float, float, float]],
    dead_frames: int,
    valid_fraction: float,
    dead_zone_d: float = 1.25,
) -> Finding:
    n = len(estimates)
    if valid_fraction < 0.4 and n < 3:
        return Finding(
            module="photorefraction",
            summary="Could not measure the eye's red reflex.",
            tier="inconclusive",
            retakes=[
                "Retake in a dim room with the screen flash step, holding still at arm's length.",
            ],
        )
    if n < 3 and dead_frames >= 3:
        return Finding(
            module="photorefraction",
            summary=(
                "No crescent visible — refractive error is likely within the "
                f"±{dead_zone_d:.1f} D dead zone of this setup (screening estimate)."
            ),
            tier="weak-signal",
            metrics={"dead_zone_d": dead_zone_d, "frames": dead_frames},
        )
    if n < 3:
        return Finding(
            module="photorefraction",
            summary="Too few usable reflex frames to estimate refraction.",
            tier="inconclusive",
            retakes=["Retake the flash step in a darker room."],
        )

    spheres = [e[0] for e in estimates]
    cyls = [e[1] for e in estimates]
    axes = [e[2] for e in estimates]
    s_med = statistics.median(spheres)
    s_std = statistics.pstdev(spheres)
    consistent = s_std <= 0.75
    tier = "measured" if (valid_fraction >= 0.7 and n >= 5 and consistent) else "weak-signal"
    summary = (
        f"Estimated defocus {s_med:+.2f} D sphere, {statistics.median(cyls):.2f} D cylinder "
        f"(screening estimate relative to camera distance)."
    )
    return Finding(
        module="photorefraction",
        summary=summary,
        tier=tier,
        metrics={
            "sphere_d": round(s_med, 2),
            "cylinder_d": round(statistics.median(cyls), 2),
            "axis_deg": round(statistics.median(axes), 1),
            "sphere_std": round(s_std, 2),
            "frames": n,
        },
    )
