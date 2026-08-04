"""Astigmatic fan / clock-dial screening.

A radial sunburst of equally spaced spokes is presented slightly fogged. An
astigmatic eye focuses one meridian on the retina and the perpendicular one
in front/behind, so ONE spoke looks sharper and darker than the rest. The
classical clinical shortcut is the "rule of 30": the clock hour that appears
darkest, multiplied by 30, gives the MINUS-cylinder axis. Equivalently, the
minus axis is perpendicular to the darkest spoke's own angle.

This test yields an AXIS, not a power — it screens for the presence and
orientation of astigmatism, complementing photorefraction (which estimates
magnitude) with an independent, purely psychophysical channel.
"""
from __future__ import annotations

import math
import statistics

from visionscreen.report import Finding

MIN_RESPONSES = 2
CONSISTENCY_DEG = 25.0   # spread across repeats to call the axis reliable


def dial_spoke_angles(n: int = 12) -> list[float]:
    """Spoke orientations in degrees, evenly spanning 180 (a line has period 180)."""
    return [round(i * 180.0 / n, 4) for i in range(n)]


def minus_cyl_axis_from_dark_meridian(dark_meridian_deg: float) -> float:
    """Minus-cylinder axis is perpendicular to the darkest (most in-focus) spoke."""
    return (dark_meridian_deg + 90.0) % 180.0


def _circular_mean_180(angles: list[float]) -> float:
    """Mean of axis-like angles (period 180), via doubled-angle vectors."""
    xs = sum(math.cos(math.radians(2 * a)) for a in angles)
    ys = sum(math.sin(math.radians(2 * a)) for a in angles)
    return (math.degrees(math.atan2(ys, xs)) / 2.0) % 180.0


def _circular_spread_180(angles: list[float]) -> float:
    mean = _circular_mean_180(angles)
    devs = []
    for a in angles:
        d = abs(a - mean) % 180.0
        devs.append(min(d, 180.0 - d))
    return statistics.mean(devs) if devs else 0.0


def score_astigmatic_dial(
    responses: list[float],
    valid_fraction: float,
    no_preference: bool,
) -> Finding:
    """responses: darkest-spoke angle (deg) reported on each repeat."""
    if valid_fraction < 0.4:
        return Finding(
            module="astigmatism",
            summary="Astigmatic dial responses could not be assessed.",
            tier="inconclusive",
            retakes=["Repeat the sunburst dial test, viewing with one eye at a time."],
        )
    if no_preference:
        return Finding(
            module="astigmatism",
            summary="All spokes of the dial appeared uniform — no meridian preference, "
                    "which argues against significant uncorrected astigmatism.",
            tier="measured",
            metrics={"flags": [], "axis_deg": None},
        )
    if len(responses) < MIN_RESPONSES:
        return Finding(
            module="astigmatism",
            summary="Not enough dial responses to determine an axis.",
            tier="inconclusive",
            retakes=["Repeat the sunburst dial test for each eye."],
        )

    dark_mean = _circular_mean_180(responses)
    spread = _circular_spread_180(responses)
    axis = minus_cyl_axis_from_dark_meridian(dark_mean)
    consistent = spread <= CONSISTENCY_DEG

    flags = ["possible astigmatism"] if consistent else []
    tier = "measured" if (consistent and valid_fraction >= 0.7) else "weak-signal"
    summary = (
        f"One meridian appeared consistently sharper — suggests astigmatism near "
        f"axis {axis:.0f}° (minus-cylinder convention)."
        if consistent
        else "Meridian reports were inconsistent between repeats; astigmatism could "
             "not be localized to an axis."
    )
    return Finding(
        module="astigmatism",
        summary=summary,
        tier=tier,
        metrics={
            "flags": flags,
            "axis_deg": round(axis, 1),
            "dark_meridian_deg": round(dark_mean, 1),
            "response_spread_deg": round(spread, 1),
            "responses": len(responses),
        },
    )
