from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from visionscreen.report import Finding
from visionscreen.synth.eyes2d import HVID_MM

# Hirschberg ratio: ~18 prism diopters of deviation per mm of corneal-reflex
# decentration (clinical range 15-22 PD/mm; see writeup for citations).
HIRSCHBERG_PD_PER_MM = 18.0
ASYMMETRY_FLAG_MM = 1.0     # ≈18 PD
ASYMMETRY_WEAK_MM = 0.5     # ≈9 PD
CONJUGACY_FLAG = 0.8
MIN_PURSUIT_SAMPLES = 20


@dataclass(frozen=True)
class AlignmentFrame:
    dec_left_mm: tuple[float, float]
    dec_right_mm: tuple[float, float]


@dataclass(frozen=True)
class PursuitResult:
    corr_left: float
    corr_right: float
    conjugacy: float


def reflex_decentration_mm(
    reflex_xy_px: tuple[float, float],
    iris_center_px: tuple[float, float],
    iris_diameter_px: float,
) -> tuple[float, float]:
    px_per_mm = iris_diameter_px / HVID_MM
    return (
        (reflex_xy_px[0] - iris_center_px[0]) / px_per_mm,
        (reflex_xy_px[1] - iris_center_px[1]) / px_per_mm,
    )


def hirschberg_pd(decentration_mm: tuple[float, float]) -> float:
    return float(np.hypot(*decentration_mm) * HIRSCHBERG_PD_PER_MM)


def pursuit_conjugacy(
    gaze_left: list[float], gaze_right: list[float], dot_x: list[float]
) -> PursuitResult | None:
    n = min(len(gaze_left), len(gaze_right), len(dot_x))
    if n < MIN_PURSUIT_SAMPLES:
        return None
    gl, gr, d = (np.asarray(s[:n], float) for s in (gaze_left, gaze_right, dot_x))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return PursuitResult(corr(gl, d), corr(gr, d), corr(gl, gr))


def _tier(valid_fraction: float) -> str:
    if valid_fraction >= 0.7:
        return "measured"
    if valid_fraction >= 0.4:
        return "weak-signal"
    return "inconclusive"


def score_alignment(
    per_frame: list[AlignmentFrame],
    pursuit: PursuitResult | None,
    valid_fraction: float,
) -> Finding:
    tier = _tier(valid_fraction)
    if tier == "inconclusive" or not per_frame:
        return Finding(
            module="alignment",
            summary="Too few usable frames to assess eye alignment.",
            tier="inconclusive",
            retakes=["Re-record the dot-following test with your face well lit and steady."],
        )

    med_l = tuple(
        statistics.median(f.dec_left_mm[i] for f in per_frame) for i in (0, 1)
    )
    med_r = tuple(
        statistics.median(f.dec_right_mm[i] for f in per_frame) for i in (0, 1)
    )
    asym_vec = (med_l[0] - med_r[0], med_l[1] - med_r[1])
    asym_mm = float(np.hypot(*asym_vec))
    deviation_pd = round(asym_mm * HIRSCHBERG_PD_PER_MM, 1)

    flags: list[str] = []
    notes: list[str] = []
    if asym_mm >= ASYMMETRY_FLAG_MM:
        flags.append("possible eye misalignment")
    elif asym_mm >= ASYMMETRY_WEAK_MM:
        notes.append("borderline reflex asymmetry")

    if pursuit is not None and pursuit.conjugacy < CONJUGACY_FLAG:
        flags.append("poor pursuit conjugacy")

    if flags:
        summary = (
            "Alignment signals observed: " + ", ".join(flags)
            + f" (Hirschberg estimate {deviation_pd} PD)."
        )
    elif notes:
        summary = f"Borderline reflex asymmetry ({deviation_pd} PD) — likely within normal range."
    else:
        summary = "No sign of eye misalignment in reflex symmetry or pursuit."

    metrics = {
        "flags": flags,
        "deviation_pd": deviation_pd,
        "asymmetry_mm": round(asym_mm, 2),
    }
    if pursuit is not None:
        metrics["conjugacy"] = round(pursuit.conjugacy, 3)
    return Finding(module="alignment", summary=summary, tier=tier, metrics=metrics)
