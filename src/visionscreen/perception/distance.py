"""Objective viewing-distance estimation from interocular pixel span.

Self-reported distance is the largest real-world error source in screen-based
acuity: because logMAR is a log-scale measure, a distance error of factor k
biases acuity by exactly log10(k). Sitting at 32 cm while claiming 50 cm makes
vision look 0.19 logMAR better than it is — nearly two chart lines, and more
than the entire algorithmic error budget.

A pinhole camera gives distance directly:  d = f · IOD / span_px.  The focal
length in pixels is obtained once, during setup, from the user's stated
distance; thereafter distance is tracked per frame and any drift is measured
rather than assumed away.
"""
from __future__ import annotations

import math
import statistics

from visionscreen.report import Finding

# Adult interpupillary distance, population mean ~63.4 mm, SD ~3.5 mm
# (ANSUR, n=3976) -> coefficient of variation ~5.5%.
DEFAULT_INTEROCULAR_MM = 63.0

# Horizontal visible iris diameter, 11.71 +/- 0.42 mm (Rufer 2005, n=743 eyes)
# -> CV 3.6%. Preferred over interocular distance for two reasons: lower
# population variance, and geometric robustness — a head yaw of theta
# foreshortens the interocular segment by cos(theta) (20 deg -> 6% distance
# error) while the iris is a circle whose projected MAJOR axis is unchanged.
DEFAULT_IRIS_MM = 11.71
DRIFT_FLAG_RATIO = 0.12      # peak-to-peak / median
MISMATCH_FLAG_RATIO = 0.10   # vs the distance the user entered
MIN_SAMPLES = 10


def estimate_focal_px(
    interocular_px: float,
    distance_mm: float,
    interocular_mm: float = DEFAULT_INTEROCULAR_MM,
) -> float | None:
    """Pinhole focal length in pixels from one known (span, distance) pair."""
    if interocular_px <= 0 or distance_mm <= 0 or interocular_mm <= 0:
        return None
    return float(interocular_px * distance_mm / interocular_mm)


def distance_from_interocular(
    interocular_px: float,
    focal_px: float,
    interocular_mm: float = DEFAULT_INTEROCULAR_MM,
) -> float | None:
    if interocular_px <= 0 or focal_px <= 0:
        return None
    return float(focal_px * interocular_mm / interocular_px)


def estimate_focal_px_from_iris(
    iris_diameter_px: float, distance_mm: float, iris_mm: float = DEFAULT_IRIS_MM
) -> float | None:
    return estimate_focal_px(iris_diameter_px, distance_mm, iris_mm)


def distance_from_iris(
    iris_diameter_px: float, focal_px: float, iris_mm: float = DEFAULT_IRIS_MM
) -> float | None:
    """Preferred ranging signal: yaw-invariant and lower-variance than IOD."""
    return distance_from_interocular(iris_diameter_px, focal_px, iris_mm)


def acuity_bias_logmar(actual_mm: float, assumed_mm: float) -> float:
    """logMAR error induced by testing at `actual` while assuming `assumed`.

    Optotype size was chosen for `assumed`; at `actual` it subtends a different
    angle, shifting the measured threshold by log10(assumed/actual).
    """
    if actual_mm <= 0 or assumed_mm <= 0:
        return 0.0
    return float(math.log10(assumed_mm / actual_mm))


def score_distance_stability(distances_mm: list[float], nominal_mm: float) -> Finding:
    n = len(distances_mm)
    if n < MIN_SAMPLES:
        return Finding(
            module="viewing distance",
            summary="Viewing distance could not be tracked from the video.",
            tier="inconclusive",
            retakes=["Re-record with your whole face visible throughout the test."],
        )

    median = statistics.median(distances_mm)
    spread = (max(distances_mm) - min(distances_mm)) / median if median else 0.0
    bias = acuity_bias_logmar(median, nominal_mm)

    flags: list[str] = []
    if spread > DRIFT_FLAG_RATIO:
        flags.append("viewing distance changed during the test")
    if median < nominal_mm * (1 - MISMATCH_FLAG_RATIO):
        flags.append("sat closer than the distance you entered")
    elif median > nominal_mm * (1 + MISMATCH_FLAG_RATIO):
        flags.append("sat farther than the distance you entered")

    if flags:
        summary = (
            f"Measured viewing distance {median/10:.0f} cm versus the "
            f"{nominal_mm/10:.0f} cm entered. This shifts acuity results by about "
            f"{abs(bias):.2f} logMAR, so treat acuity as approximate."
        )
    else:
        summary = (
            f"Viewing distance held steady at about {median/10:.0f} cm, matching "
            "the entered value."
        )

    return Finding(
        module="viewing distance",
        summary=summary,
        tier="measured",
        metrics={
            "flags": flags,
            "median_cm": round(median / 10, 1),
            "spread_pct": round(100 * spread, 1),
            "acuity_bias_logmar": round(bias, 3),
            "frames": n,
        },
    )
