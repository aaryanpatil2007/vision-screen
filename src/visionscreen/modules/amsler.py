"""Amsler grid: central 20-degree field screening for macular disease.

Standard chart (Amsler 1947): a 10 cm square of 5 mm boxes viewed at 33 cm,
subtending 20 degrees total (10 degrees eccentricity each way), each box
1 degree, with a central fixation dot. Tested monocularly — the fellow eye
fills in defects binocularly, which is exactly why patients miss their own
scotomas until late.

Positive findings (wavy/distorted lines = metamorphopsia, missing patches =
scotoma) warrant prompt referral, so this module escalates its wording rather
than reporting a bland flag.
"""
from __future__ import annotations

import math

from visionscreen.report import Finding

AMSLER_SQUARES = 20
AMSLER_SIDE_CM = 10.0
AMSLER_DISTANCE_CM = 33.0


def amsler_geometry(
    distance_cm: float, px_per_cm: float, target_deg: float = 20.0
) -> dict:
    """Grid size in px that subtends the standard 20 degrees at this distance.

    The textbook "10 cm at 33 cm" is a rounded figure (it is really 17.2 deg);
    on a screen the physical size is ours to choose, so we solve for the size
    that reproduces the intended angular subtense exactly at whatever viewing
    distance the session is using.
    """
    side_cm = 2 * distance_cm * math.tan(math.radians(target_deg / 2))
    return {
        "total_deg": target_deg,
        "square_deg": target_deg / AMSLER_SQUARES,
        "squares": AMSLER_SQUARES,
        "grid_px": side_cm * px_per_cm,
        "grid_cm": side_cm,
        "distance_cm": distance_cm,
    }


def score_amsler(
    distortions: list[dict],
    missing: list[dict],
    eyes_tested: int,
    valid_fraction: float,
) -> Finding:
    if valid_fraction < 0.4:
        return Finding(
            module="amsler",
            summary="Amsler grid test could not be assessed.",
            tier="inconclusive",
            retakes=["Repeat the grid test, covering one eye at a time."],
        )

    flags: list[str] = []
    if distortions:
        flags.append("distorted lines (metamorphopsia)")
    if missing:
        flags.append("missing area (possible scotoma)")

    if flags:
        summary = (
            "You marked " + " and ".join(flags) + " on the central grid. "
            "Findings like these can indicate a macular problem and should be "
            "checked by an optometrist or ophthalmologist promptly."
        )
    else:
        summary = "Central 20-degree grid appeared complete and undistorted in both eyes."

    tier = "measured" if (eyes_tested >= 2 and valid_fraction >= 0.7) else "weak-signal"
    if eyes_tested < 2 and not flags:
        summary += " Only one eye was tested, so a one-sided defect could be missed."

    return Finding(
        module="amsler",
        summary=summary,
        tier=tier,
        metrics={
            "flags": flags,
            "distortion_marks": len(distortions),
            "missing_marks": len(missing),
            "eyes_tested": eyes_tested,
        },
    )
