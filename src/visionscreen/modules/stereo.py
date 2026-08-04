"""Stereoacuity from a dynamic random-dot stereogram.

**Why random dots and not circles.** Contour stereotests leak badly. With the
plate rotated 90 degrees — removing all valid stereo information while
preserving monocular offsets — every subject tested still "passed" the coarse
levels of a Titmus-type circles test and 27% reached the finest 40 arcsec rung
(Hartle, Vancleef, Read et al., Sci Rep 2019). The leak is a *binocular
non-stereoscopic* lateral-displacement cue, so covering one eye does not
screen it out. This module therefore uses dynamic random-dot stereograms with
per-frame dot re-randomization, four-alternative forced choice, and
**zero-disparity catch trials**: if a subject "passes" trials that contain no
depth information, the entire result is voided rather than reported.

**Display floor.** On-screen disparity is quantized by pixel pitch:
theta = 206265 * d / D arcsec. A 140 ppi laptop at 50 cm has a whole-pixel
floor near 75 arcsec and simply cannot present the 40 arcsec rung. Antialiased
subpixel rendering buys a measured 2-4x (ASTEROID reached 29.5-56 arcsec below
a 118 arcsec whole-pixel floor), but the floor is reported alongside the
result so a floor-limited answer is never mistaken for a measured one.

**Tier ceiling.** No browser stereotest of this design has published agreement
against a clinical stereotest, so results cap at `weak-signal`.
"""
from __future__ import annotations

from visionscreen.report import Finding

ARCSEC_PER_RADIAN = 206265.0
NORMAL_ARCSEC = 60.0        # adults typically reach 20-40; >60 is worth noting
REDUCED_ARCSEC = 200.0
MIN_TRIALS = 6
CATCH_PASS_LIMIT = 0.6      # above chance (0.25 for 4AFC) by a clear margin


def disparity_arcsec(pixel_disparity_mm: float, distance_mm: float) -> float:
    """Screen disparity in arcsec. Independent of interpupillary distance,
    and scaling as 1/D (real-depth tests scale as 1/D^2)."""
    if distance_mm <= 0:
        return 0.0
    return ARCSEC_PER_RADIAN * pixel_disparity_mm / distance_mm


def min_resolvable_arcsec(
    px_per_cm: float, distance_cm: float, subpixel_factor: float = 1.0
) -> float:
    """Finest disparity this display can present at this distance."""
    if px_per_cm <= 0 or distance_cm <= 0:
        return float("inf")
    pitch_mm = 10.0 / px_per_cm
    return disparity_arcsec(pitch_mm, distance_cm * 10.0) / max(subpixel_factor, 1.0)


def score_stereo(
    trials: list[dict],
    catch_trials: list[dict],
    valid_fraction: float,
    display_floor_arcsec: float | None = None,
) -> Finding:
    """trials: [{"arcsec": float, "correct": bool}]; catch_trials carry no disparity."""
    if valid_fraction < 0.4 or len(trials) < MIN_TRIALS:
        return Finding(
            module="stereo",
            summary="Depth-perception test was not completed.",
            tier="inconclusive",
            retakes=["Complete the 3-D dot test wearing the red-cyan glasses."],
        )

    # Catch trials first: a subject reading a non-stereo cue invalidates everything.
    if catch_trials:
        passed = sum(1 for c in catch_trials if c.get("correct")) / len(catch_trials)
        if passed > CATCH_PASS_LIMIT:
            return Finding(
                module="stereo",
                summary=(
                    "Depth answers were correct even on patterns containing no depth "
                    "information, so the responses were driven by a non-stereo cue "
                    "(a brightness or position artifact) rather than binocular vision. "
                    "No stereo result can be reported from this run."
                ),
                tier="inconclusive",
                metrics={"catch_pass_rate": round(passed, 2), "flags": []},
                retakes=[
                    "Check the red-cyan glasses are on the correct way round and repeat.",
                    "Dim reflections on the screen — glare lets one eye see the other's image.",
                ],
            )

    by_level: dict[float, list[bool]] = {}
    for t in trials:
        by_level.setdefault(float(t["arcsec"]), []).append(bool(t["correct"]))

    # threshold = finest (smallest arcsec) level still reliably correct
    threshold = max(by_level)
    for level in sorted(by_level, reverse=True):
        results = by_level[level]
        if sum(results) / len(results) >= 0.6:
            threshold = level
        else:
            break

    flags: list[str] = []
    if threshold > REDUCED_ARCSEC:
        flags.append("reduced stereo depth perception")
    elif threshold > NORMAL_ARCSEC:
        flags.append("borderline stereo depth perception")

    floor_limited = (
        display_floor_arcsec is not None and threshold <= display_floor_arcsec * 1.05
    )
    if floor_limited:
        summary = (
            f"Depth perception at or better than {threshold:.0f} arcseconds — the finest "
            f"this display can present at your viewing distance, so true stereoacuity "
            "may be better."
        )
    elif flags:
        summary = (
            f"Stereo depth threshold {threshold:.0f} arcseconds — {flags[0]}. "
            "Reduced stereo vision can accompany eye misalignment or amblyopia."
        )
    else:
        summary = f"Normal stereo depth perception ({threshold:.0f} arcseconds)."

    metrics = {
        "threshold_arcsec": round(threshold, 1),
        "flags": flags,
        "trials": len(trials),
        "catch_trials": len(catch_trials),
    }
    if display_floor_arcsec is not None:
        metrics["display_floor_arcsec"] = round(display_floor_arcsec, 1)
    return Finding(
        module="stereo",
        summary=summary + " Screen-based stereo testing is not calibrated against "
                          "clinical stereotests; treat as a rough screen.",
        tier="weak-signal",
        metrics=metrics,
    )
