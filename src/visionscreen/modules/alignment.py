from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from visionscreen.report import Finding
from visionscreen.synth.eyes2d import HVID_MM

# Hirschberg ratio: ~18 prism diopters of deviation per mm of corneal-reflex
# decentration (clinical range 15-22 PD/mm; see writeup for citations).
HIRSCHBERG_PD_PER_MM = 18.0
# Clinically significant strabismus starts around 10 prism diopters, and our
# measurement error is ~0.6 PD (bench_module2), so flagging at 10 PD costs
# almost nothing in false positives while catching the 10-18 PD deviations an
# 18 PD threshold silently missed (battery benchmark: sensitivity 0.73 -> 0.99).
ASYMMETRY_FLAG_MM = 10.0 / HIRSCHBERG_PD_PER_MM    # ≈0.56 mm
ASYMMETRY_WEAK_MM = 5.0 / HIRSCHBERG_PD_PER_MM     # ≈0.28 mm
CONJUGACY_FLAG = 0.8
MIN_PURSUIT_SAMPLES = 20

# Physiological ceiling. Human horizontal deviations essentially never exceed
# ~90 PD, and validation on real uncontrolled photographs produced readings up
# to 163 PD on normally-aligned faces — those are stray specular highlights
# (windows, lamps, spectacle glare) being mistaken for the corneal catchlight,
# not measurements. Above this bound the pair is not a usable Hirschberg
# observation, so no magnitude is reported.
MAX_PLAUSIBLE_PD = 60.0

# A true deviation is stable across frames; uncorrelated stray highlights
# jitter. Require enough agreement before quoting a number.
#
# The frame count is set from measurement, not taste. On real eye crops with a
# glint injected at a known decentration (bench_real_hirschberg), single-frame
# error is 5.35 PD mean absolute, implying a per-frame sigma of 6.7 PD. The
# median of n frames then carries roughly 1.25*sigma/sqrt(n) of standard error:
#
#     n = 5  -> 3.0 PD      n = 20 -> 1.5 PD      n = 40 -> 1.1 PD
#
# and the measured value at n = 40 was 1.09 PD, matching the model. Twenty
# frames keeps the magnitude error near 1.5 PD, comfortably inside the ~5 PD
# interexaminer agreement of the prism cover test itself, so a 10 PD flag is
# meaningful. Below that the number is not worth quoting.
MIN_FRAMES_FOR_MAGNITUDE = 20
PER_FRAME_SIGMA_PD = 6.7        # measured on real eye images
MAX_ASYMMETRY_DISPERSION_MM = 0.45


def expected_magnitude_error_pd(n_frames: int) -> float:
    """Expected mean absolute error of the aggregated deviation, in PD.

    The 1.253 factor is the asymptotic efficiency of the median relative to the
    mean; it does not apply at n = 1, where the median simply is the sample.
    """
    if n_frames < 1:
        return float("inf")
    import math
    efficiency = 1.0 if n_frames == 1 else 1.253
    se = efficiency * PER_FRAME_SIGMA_PD / math.sqrt(n_frames)
    return se * math.sqrt(2 / math.pi)


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
    if d.std() < 1e-6:
        return None  # the dot never moved — pursuit is unmeasurable, not bad

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
        # Hirschberg needs the corneal reflex; pursuit only needs iris tracking.
        # A real webcam often loses the reflex — report pursuit on its own.
        if pursuit is not None:
            flags = ["poor pursuit conjugacy"] if pursuit.conjugacy < CONJUGACY_FLAG else []
            summary = (
                ("Pursuit conjugacy is reduced — eyes did not track together. "
                 if flags else "Both eyes tracked the target together normally. ")
                + "Corneal reflex was not measurable, so the Hirschberg check was skipped "
                "(try again with a lamp in front of you)."
            )
            return Finding(
                module="alignment",
                summary=summary,
                tier="weak-signal",
                metrics={"flags": flags, "conjugacy": round(pursuit.conjugacy, 3)},
            )
        return Finding(
            module="alignment",
            summary=(
                "Eye alignment could not be measured: no corneal light reflection was "
                "visible. In ordinary room lighting only about one webcam frame in ten "
                "shows one, which is why this test uses a bright target on a dark screen."
            ),
            tier="inconclusive",
            retakes=[
                "Dim the room lights and repeat the dot-following test — the bright dot "
                "needs to be the strongest light on your face.",
                "Remove glasses if you can; lens reflections hide the corneal glint.",
            ],
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

    # --- artifact rejection, learned from real-photograph validation ---
    per_frame_asym = [
        float(np.hypot(f.dec_left_mm[0] - f.dec_right_mm[0],
                       f.dec_left_mm[1] - f.dec_right_mm[1]))
        for f in per_frame
    ]
    dispersion = (
        statistics.pstdev(per_frame_asym) if len(per_frame_asym) > 1 else 0.0
    )
    implausible = deviation_pd > MAX_PLAUSIBLE_PD
    unstable = (
        len(per_frame) >= MIN_FRAMES_FOR_MAGNITUDE
        and dispersion > MAX_ASYMMETRY_DISPERSION_MM
    )
    too_few = len(per_frame) < MIN_FRAMES_FOR_MAGNITUDE

    if implausible or unstable:
        return Finding(
            module="alignment",
            summary=(
                "The bright spots used to judge alignment were not consistent "
                "between the eyes or between frames, which happens when a lamp, "
                "window or spectacle lens reflects into the camera. No alignment "
                "measurement can be trusted from this recording."
            ),
            tier="inconclusive",
            metrics={
                "flags": [],
                "rejected_reason": "implausible magnitude" if implausible
                                   else "unstable across frames",
                "frames": len(per_frame),
            },
            retakes=[
                "Repeat in a dim room so the on-screen target is the brightest "
                "light on your face.",
                "Remove glasses if you can — lens reflections imitate the corneal glint.",
            ],
        )

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
        "frames": len(per_frame),
        "expected_error_pd": round(expected_magnitude_error_pd(len(per_frame)), 1),
    }
    if len(per_frame) > 1:
        metrics["asymmetry_dispersion_mm"] = round(dispersion, 3)
    if pursuit is not None:
        metrics["conjugacy"] = round(pursuit.conjugacy, 3)

    # A magnitude from one or two frames is not a measurement, whatever the
    # capture quality: a single stray highlight cannot be distinguished from a
    # real deviation without frame-to-frame agreement.
    if too_few:
        tier = "weak-signal"
        summary += (
            f" Based on only {len(per_frame)} usable frame"
            f"{'s' if len(per_frame) != 1 else ''}, so the magnitude is provisional."
        )
    return Finding(module="alignment", summary=summary, tier=tier, metrics=metrics)
