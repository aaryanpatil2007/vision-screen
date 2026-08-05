"""What the front of the eye shows: media opacity, sclera colour, lid position.

The rest of the battery asks the person questions. This module just looks at
them. Everything here is measured from a colour eye crop plus the sclera / iris
/ pupil masks the segmentation network produces, so it costs the user nothing
beyond frames that are already being captured.

Which of these a webcam can honestly support divides cleanly in two:

**Geometry is reliable.** Lid margin position, pupil ratio, the shape of the
iris boundary — these survive bad lighting, bad white balance, and cheap
optics, because they are ratios of distances within one frame, calibrated
against the horizontal visible iris diameter (11.71 +/- 0.42 mm, Rufer 2005),
which is one of the most stable dimensions in the human body.

**Colour is not.** Webcam auto-white-balance will happily turn a jaundiced
sclera neutral, or a neutral sclera yellow under tungsten light. An absolute
claim about sclera colour from an uncalibrated consumer camera is not
supportable, and the published smartphone work that does make such claims
(BiliScreen and its successors) uses a physical colour reference in frame for
exactly this reason.

So this module takes the cheap version of that fix: it asks the user to hold a
plain sheet of white paper beside their face for one frame. Paper is
free and universal, and it turns an absolute colour question into a relative
one. Without it, colour findings are still computed but are capped at
weak-signal and labelled as needing proper conditions — they are never
promoted to a finding someone might act on.

The red-reflex analysis is the weakest thing here and is capped accordingly.
It is comparative by construction, which does spare it the white-balance
problem — but a laptop has no flash, and without one there may simply be no
reflex to compare. See `REFLEX_MAX_TIER` for why it can never report more than
a weak signal, and what happened to the one app that tried.

Ranked by how much a webcam can actually support them, the findings here go:
lid position (geometry, published AUC 0.94 from mobile photos) > iris boundary
> corneal arcus > sclera colour (needs the white reference, and thresholds are
not yet calibrated) > red reflex (capped, cannot rule anything out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from visionscreen.report import Finding

HVID_MM = 11.71            # horizontal visible iris diameter, Rufer 2005
HVID_SD_MM = 0.42

# --- red reflex / Brueckner -------------------------------------------------
# A cataract or other media opacity scatters and absorbs the light coming back
# out of the eye, so the reflex is dimmer, patchier, or partly shadowed. The
# clinically meaningful quantity is the difference between the two eyes: a
# unilateral opacity is both far more likely to be pathological and far more
# detectable than a symmetric one.
REFLEX_ASYMMETRY_FLAG = 0.35      # fractional brightness difference between eyes
REFLEX_NONUNIFORMITY_FLAG = 0.28  # within-pupil coefficient of variation
LEUKOCORIA_SATURATION_MAX = 0.20  # a white (not red/orange) reflex
LEUKOCORIA_VALUE_MIN = 0.65

# The red-reflex analysis below is deliberately capped, and this constant is
# where that is enforced. The optics do not support more.
#
# A Brueckner test needs illumination close to coaxial with the lens and bright
# enough to light the fundus through an undilated pupil. A laptop has no flash:
# the "screen flash" this system uses is broad, dim, offset from the camera by
# several centimetres, and filtered by the webcam's own infrared-cut filter,
# which rejects exactly the wavelengths that return most strongly from the
# fundus. Dedicated photoscreeners use an infrared source at a controlled
# eccentricity for these reasons.
#
# The empirical warning is stark. CRADLE, a leukocoria-detection app whose
# developers reported 90% per-child sensitivity, achieved 15.4% sensitivity in
# independent prospective validation (Vagge 2019) -- and it used a real camera
# flash for essentially every positive image. A separate evaluation found it
# "failed to provide adequate leukocoria detection except four late stage"
# cases. Cataract grading (LOCS III) is defined on slit-lamp and
# retroillumination views; that is an optical fact, not a missing dataset.
#
# So this module reports gross between-eye asymmetry and nothing more, never
# above weak-signal, and says plainly what it cannot do. A screening test that
# looks like a cataract detector and misses six in seven cases is worse than no
# test, because it converts "I should get my eyes checked" into false
# reassurance.
REFLEX_MAX_TIER = "weak-signal"
REFLEX_CANNOT_DO = (
    "This check cannot rule anything out. A laptop camera has no flash and "
    "filters out the light that would show the reflex properly, so it sees far "
    "less than the handheld light an examiner uses — purpose-built versions of "
    "this test have missed most real cases in independent testing. Treat a "
    "clear result here as meaning nothing was visible, not as an all-clear."
)

# --- lid position -----------------------------------------------------------
# MRD1: upper lid margin to the centre of the pupil, with the eye in primary
# gaze. Normal is about 4-5 mm; below 2 mm is the usual surgical threshold for
# ptosis. Measured in iris-diameters and converted with HVID.
MRD1_PTOSIS_MM = 2.0
MRD1_NORMAL_MM = 4.0
MRD1_ASYMMETRY_MM = 1.5

# --- sclera colour ----------------------------------------------------------
# CIELAB b* over the exposed sclera, and a redness index (R-G)/(R+G).
#
# These two numbers are the least defensible constants in this module and are
# labelled as such. Everything else here is either a physical dimension (HVID),
# a published clinical threshold (MRD1 < 2 mm for ptosis), or a within-frame
# comparison that needs no absolute reference (reflex asymmetry). These are
# neither: they are plausible cut-points chosen to separate an obviously yellow
# or obviously red eye from a normal one, with no labelled corpus behind them.
#
# The consequence is enforced in code rather than left to a reader's judgement:
# `SCLERA_COLOUR_CALIBRATED` is False, and while it is False `score_sclera`
# refuses to emit a "measured" tier no matter how good the white balance is.
# Wiring a validated threshold in means setting this flag, and the test suite
# checks that an uncalibrated build cannot produce an actionable colour claim.
ICTERUS_B_STAR = 18.0
INJECTION_REDNESS = 0.30          # (R - G) / (R + G) over the sclera
SCLERA_COLOUR_CALIBRATED = False  # no labelled corpus behind the two above

# --- corneal arcus ----------------------------------------------------------
# A lipid ring at the corneal periphery. Shows up as a brighter, desaturated
# annulus just inside the limbus, separated from it by a clear zone.
ARCUS_CONTRAST = 0.18


@dataclass
class EyeAppearance:
    """Raw measurements from one eye; interpretation happens in the scorers."""

    side: str
    reflex_mean: float | None = None
    reflex_cv: float | None = None
    reflex_saturation: float | None = None
    reflex_value: float | None = None
    mrd1_mm: float | None = None
    sclera_b_star: float | None = None
    sclera_redness: float | None = None
    arcus_contrast: float | None = None
    iris_circularity: float | None = None
    nasal_encroachment: float | None = None
    px_per_mm: float | None = None


# ------------------------------------------------------------- calibration --

def px_per_mm_from_iris(iris_mask: np.ndarray) -> float | None:
    """Scale from the one dimension in the eye that barely varies between people.

    Uses the horizontal extent of the iris mask rather than a fitted circle,
    because the upper and lower iris are usually occluded by the lids while the
    horizontal extent is not.
    """
    cols = np.where(iris_mask.any(axis=0))[0]
    if cols.size < 8:
        return None
    return float(cols.max() - cols.min() + 1) / HVID_MM


def white_balance(img: np.ndarray, reference_rgb: tuple[float, float, float] | None
                  ) -> tuple[np.ndarray, bool]:
    """Scale channels so a known-white patch reads neutral.

    Returns (image, calibrated). When no reference is supplied the image is
    returned untouched and `calibrated` is False, which every colour scorer
    below uses to cap its own confidence rather than silently guessing.
    """
    if reference_rgb is None:
        return img, False
    ref = np.asarray(reference_rgb, np.float64)
    if ref.min() <= 1.0 or ref.max() >= 254.0:
        # a clipped or black reference carries no white-balance information
        return img, False
    gain = ref.mean() / np.maximum(ref, 1.0)
    out = np.clip(img.astype(np.float64) * gain.reshape(1, 1, 3), 0, 255)
    return out.astype(np.uint8), True


# ------------------------------------------------------- red reflex metrics --

def measure_red_reflex(flash_rgb: np.ndarray, pupil_mask: np.ndarray
                       ) -> dict | None:
    """Brightness, uniformity and hue of the light coming back out of the pupil.

    Restricted to the pupil mask: the corneal glint and the iris are both far
    brighter than the reflex and would dominate any unmasked statistic. This is
    the same failure that made corneal-reflex localisation useless earlier in
    this project, so the mask is not optional.
    """
    if pupil_mask.sum() < 30:
        return None
    # the specular glint is a small, blown-out spot; excluding the top decile
    # of luminance removes it without touching the reflex itself
    gray = cv2.cvtColor(flash_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    inside = pupil_mask.astype(bool)
    vals = gray[inside]
    if vals.size < 30:
        return None
    cutoff = np.percentile(vals, 90)
    keep = inside & (gray <= cutoff)
    if keep.sum() < 20:
        keep = inside
    lit = gray[keep]

    hsv = cv2.cvtColor(flash_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    sat = hsv[..., 1][keep] / 255.0
    val = hsv[..., 2][keep] / 255.0

    mean = float(lit.mean())
    return {
        "mean": mean / 255.0,
        "cv": float(lit.std() / max(mean, 1e-6)),
        "saturation": float(sat.mean()),
        "value": float(val.mean()),
        "pixels": int(keep.sum()),
    }


def score_red_reflex(left: dict | None, right: dict | None,
                     valid_fraction: float = 1.0) -> Finding:
    """The Brueckner test, as far as a webcam can take it.

    Deliberately conservative about what it will name. A dim or patchy reflex
    has many innocent causes on a consumer camera — off-axis gaze, a partly
    closed lid, a dirty lens, an eyelash — and exactly one alarming one. The
    finding therefore describes what was seen and who should look at it, and
    never says the word it would be irresponsible to say from this evidence.
    """
    if not left and not right:
        return Finding(
            module="red reflex",
            summary="The reflex from the back of the eye was not captured.",
            tier="inconclusive",
            retakes=["Repeat the flash step in a dark room, looking straight at "
                     "the camera with both eyes open."],
        )
    if valid_fraction < 0.4:
        return Finding(
            module="red reflex",
            summary="Too few usable flash frames to assess the reflex.",
            tier="inconclusive",
            retakes=["Repeat the flash step, holding still."],
        )

    flags: list[str] = []
    metrics: dict = {}
    notes: list[str] = []

    # --- leukocoria: a white rather than red/orange reflex ------------------
    # This is the one finding here that can be an emergency (retinoblastoma in
    # a child, advanced cataract in an adult), so it is checked per eye and
    # never requires the other eye to be present.
    for side, m in (("left", left), ("right", right)):
        if not m:
            continue
        metrics[f"{side}_reflex_brightness"] = round(m["mean"], 3)
        metrics[f"{side}_reflex_uniformity"] = round(1.0 - min(m["cv"], 1.0), 3)
        if (m["saturation"] < LEUKOCORIA_SATURATION_MAX
                and m["value"] > LEUKOCORIA_VALUE_MIN):
            flags.append(f"{side}: pale reflex")
            notes.append(
                f"The {side} eye returned a pale rather than warm-coloured "
                "reflex."
            )

    # --- asymmetry: the actual Brueckner signal ----------------------------
    if left and right:
        hi = max(left["mean"], right["mean"])
        lo = min(left["mean"], right["mean"])
        asym = (hi - lo) / max(hi, 1e-6)
        metrics["reflex_asymmetry"] = round(asym, 3)
        if asym > REFLEX_ASYMMETRY_FLAG:
            dimmer = "left" if left["mean"] < right["mean"] else "right"
            flags.append("asymmetric reflex")
            notes.append(
                f"The two eyes returned noticeably different amounts of light "
                f"({dimmer} dimmer). A difference between the eyes is the "
                "specific thing this check looks for."
            )
    else:
        notes.append("Only one eye was captured, so the two could not be compared "
                     "— which is the comparison this check depends on.")

    # --- uniformity: shadows within the reflex -----------------------------
    for side, m in (("left", left), ("right", right)):
        if m and m["cv"] > REFLEX_NONUNIFORMITY_FLAG:
            flags.append(f"{side}: patchy reflex")
            notes.append(f"The {side} eye's reflex was uneven across the pupil.")

    if not flags:
        return Finding(
            module="red reflex",
            summary=("Nothing uneven was visible in the light returning from "
                     "either eye, and the two matched each other. " + REFLEX_CANNOT_DO),
            tier=REFLEX_MAX_TIER,
            metrics=metrics,
        )

    return Finding(
        module="red reflex",
        summary=(
            " ".join(notes)
            + " This is an observation, not a diagnosis, and several harmless "
            "things imitate it — an off-centre gaze, a partly closed lid, a "
            "smudged lens. It is worth having someone look properly. "
            + REFLEX_CANNOT_DO
        ),
        tier=REFLEX_MAX_TIER,
        metrics={**metrics, "flags": flags},
        retakes=["Repeat the flash step in a fully dark room to confirm the "
                 "finding is not a lighting artefact."],
    )


# ------------------------------------------------------------ lid position --

def measure_mrd1(upper_lid_y: float, pupil_center_y: float,
                 px_per_mm: float) -> float:
    """Margin-reflex distance 1, in millimetres.

    Positive means the lid sits above the pupil centre, which is normal.
    """
    return (pupil_center_y - upper_lid_y) / max(px_per_mm, 1e-6)


def score_ptosis(left_mrd1: float | None, right_mrd1: float | None,
                 gaze_ok: bool = True) -> Finding:
    """Droop of the upper lid, measured rather than eyeballed.

    Only valid in primary gaze: looking up or down moves the lid with the eye,
    so a person glancing away produces a perfectly convincing false positive.
    """
    if left_mrd1 is None and right_mrd1 is None:
        return Finding(module="eyelid position",
                       summary="Lid position could not be measured.",
                       tier="inconclusive",
                       retakes=["Look straight at the camera with both eyes open."])
    if not gaze_ok:
        return Finding(
            module="eyelid position",
            summary=("Lid position was not assessed because the eyes were not "
                     "looking straight ahead — the lid follows the eye, so the "
                     "measurement would not have meant anything."),
            tier="inconclusive",
            retakes=["Look directly at the camera for the whole of this step."],
        )

    metrics, flags, notes = {}, [], []
    for side, v in (("left", left_mrd1), ("right", right_mrd1)):
        if v is None:
            continue
        metrics[f"{side}_mrd1_mm"] = round(v, 2)
        if v < MRD1_PTOSIS_MM:
            flags.append(f"{side}: low lid")
            notes.append(f"The {side} upper lid sits low over the pupil "
                         f"({v:.1f} mm above centre; {MRD1_NORMAL_MM:.0f} mm is typical).")

    if left_mrd1 is not None and right_mrd1 is not None:
        diff = abs(left_mrd1 - right_mrd1)
        metrics["mrd1_difference_mm"] = round(diff, 2)
        if diff > MRD1_ASYMMETRY_MM:
            flags.append("asymmetric lids")
            notes.append(f"The two lids differ by {diff:.1f} mm, which is more "
                         "than the usual difference between a person's own eyes.")

    if not flags:
        return Finding(
            module="eyelid position",
            summary="Both upper lids sat at a typical height relative to the pupil.",
            tier="measured", metrics=metrics)

    return Finding(
        module="eyelid position",
        summary=" ".join(notes) + " A drooping lid has causes ranging from the "
                "entirely benign to ones worth checking promptly, so this is "
                "worth mentioning to a clinician rather than acting on here.",
        tier="measured" if len(metrics) >= 3 else "weak-signal",
        metrics={**metrics, "flags": flags},
    )


# ---------------------------------------------------------- sclera colour ---

def measure_sclera_colour(rgb: np.ndarray, sclera_mask: np.ndarray) -> dict | None:
    """Mean CIELAB b* (yellowness) and a redness index over exposed sclera.

    Erodes the mask first: the pixels at the sclera boundary are a blend of
    sclera with iris, lid margin and lashes, and they are precisely the pixels
    that would bias both statistics in the alarming direction.
    """
    mask = cv2.erode(sclera_mask.astype(np.uint8),
                     np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    if mask.sum() < 50:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    b_star = float(lab[..., 2][mask].mean() - 128.0)

    px = rgb.astype(np.float64)[mask]
    r, g = px[:, 0], px[:, 1]
    redness = float(np.mean((r - g) / np.maximum(r + g, 1.0)))
    return {"b_star": b_star, "redness": redness,
            "luminance": float(lab[..., 0][mask].mean()),
            "pixels": int(mask.sum())}


def score_sclera(left: dict | None, right: dict | None,
                 calibrated: bool) -> Finding:
    """Yellowing and redness of the white of the eye.

    Capped at weak-signal without a white reference in frame, because an
    uncalibrated webcam cannot distinguish a yellow eye from yellow light. That
    cap is the honest position, not a hedge: the published smartphone work on
    scleral jaundice carries a physical colour card for this reason.
    """
    if not left and not right:
        return Finding(module="sclera appearance",
                       summary="The white of the eye was not clearly visible.",
                       tier="inconclusive",
                       retakes=["Look straight ahead with eyes open in even light."])

    metrics, flags, notes = {}, [], []
    for side, m in (("left", left), ("right", right)):
        if not m:
            continue
        metrics[f"{side}_yellowness"] = round(m["b_star"], 1)
        metrics[f"{side}_redness"] = round(m["redness"], 3)
        if m["b_star"] > ICTERUS_B_STAR:
            flags.append(f"{side}: yellow tinge")
        if m["redness"] > INJECTION_REDNESS:
            flags.append(f"{side}: redness")

    if not calibrated:
        base = ("Colour of the whites of the eyes was recorded but not judged: "
                "without a white reference in the picture, a camera cannot tell "
                "a yellow eye from yellow room lighting.")
        if flags:
            return Finding(
                module="sclera appearance",
                summary=base + " The raw numbers did fall outside the usual "
                        "range, which is worth re-checking properly rather than "
                        "reading anything into as it stands.",
                tier="weak-signal", metrics={**metrics, "flags": ["uncalibrated"]},
                retakes=["Repeat this step holding a plain sheet of white paper "
                         "beside your face, so the camera has a reference."])
        return Finding(module="sclera appearance", summary=base,
                       tier="inconclusive", metrics=metrics,
                       retakes=["Repeat holding a plain sheet of white paper "
                                "beside your face."])

    # The white reference fixes the camera, not the thresholds. Until the
    # cut-points above are set from labelled images, a colour reading is at
    # best a suggestion to look properly, so it stays below the tier the report
    # treats as actionable.
    colour_tier = "measured" if SCLERA_COLOUR_CALIBRATED else "weak-signal"
    provisional = "" if SCLERA_COLOUR_CALIBRATED else (
        " The colour thresholds behind this reading have not yet been set "
        "against a reference set of real eyes, so treat it as a prompt to "
        "check rather than a result.")

    if not flags:
        return Finding(
            module="sclera appearance",
            summary="The whites of the eyes were of normal colour against the "
                    "white reference." + provisional,
            tier=colour_tier, metrics=metrics)

    yellow = [f for f in flags if "yellow" in f]
    red = [f for f in flags if "redness" in f]
    parts = []
    if yellow:
        parts.append("The whites of the eyes looked yellower than usual. That "
                     "can reflect something going on with the liver or blood "
                     "rather than the eye itself, and is worth a same-week "
                     "appointment with a doctor.")
    if red:
        parts.append("The whites of the eyes looked red. Most causes are minor "
                     "and settle on their own, but redness with pain, light "
                     "sensitivity or changed vision should be seen quickly.")
    return Finding(module="sclera appearance", summary=" ".join(parts) + provisional,
                   tier=colour_tier, metrics={**metrics, "flags": flags})


# ------------------------------------------------------------ corneal arcus --

def measure_arcus(gray: np.ndarray, iris_mask: np.ndarray,
                  center: tuple[float, float], radius: float) -> float | None:
    """Contrast of a bright annulus just inside the limbus.

    Arcus is a lipid deposit at the corneal periphery; on camera it reads as a
    pale ring separated from the iris body. Compares the outermost tenth of the
    iris radius against the band just inside it.
    """
    if radius < 8:
        return None
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - center[0], yy - center[1]) / radius
    mask = iris_mask.astype(bool)
    rim = mask & (r > 0.86) & (r <= 0.98)
    body = mask & (r > 0.55) & (r <= 0.80)
    if rim.sum() < 25 or body.sum() < 25:
        return None
    g = gray.astype(np.float64)
    rim_v, body_v = g[rim].mean(), g[body].mean()
    return float((rim_v - body_v) / max(rim_v + body_v, 1e-6))


def score_arcus(left: float | None, right: float | None,
                age: float | None = None) -> Finding:
    """A pale ring at the edge of the coloured part of the eye.

    Interpretation depends almost entirely on age, which is why age is an input
    rather than a nicety: over about 60 arcus is an unremarkable finding present
    in most people, while under 45 it can point at inherited high cholesterol
    and genuinely changes what someone should do next.
    """
    vals = {s: v for s, v in (("left", left), ("right", right)) if v is not None}
    if not vals:
        return Finding(module="corneal arcus",
                       summary="The edge of the iris was not clear enough to assess.",
                       tier="inconclusive")

    metrics = {f"{s}_arcus_contrast": round(v, 3) for s, v in vals.items()}
    present = [s for s, v in vals.items() if v > ARCUS_CONTRAST]
    if not present:
        return Finding(module="corneal arcus",
                       summary="No pale ring was seen at the edge of the iris.",
                       tier="measured", metrics=metrics)

    if age is not None and age < 45:
        summary = ("A pale ring was seen at the edge of the coloured part of the "
                   "eye. Below about 45 this is worth mentioning to a doctor, "
                   "because it is sometimes associated with high blood "
                   "cholesterol — a blood test settles it.")
        tier = "weak-signal"
    elif age is not None:
        summary = ("A pale ring was seen at the edge of the coloured part of the "
                   "eye. Above about 60 this is a common age-related finding "
                   "that does not affect vision and needs nothing done.")
        tier = "measured"
    else:
        summary = ("A pale ring was seen at the edge of the coloured part of the "
                   "eye. What it means depends heavily on age — common and "
                   "harmless later in life, worth a cholesterol check earlier.")
        tier = "weak-signal"
    return Finding(module="corneal arcus", summary=summary, tier=tier,
                   metrics={**metrics, "flags": [f"{s}: arcus" for s in present]})


# --------------------------------------------------- iris boundary / growth --

def measure_iris_boundary(iris_mask: np.ndarray) -> dict | None:
    """How far the iris outline departs from a circle, and where.

    A pterygium grows across the cornea from the nasal side and occludes part of
    the iris, so the iris mask develops a wedge-shaped bite out of one side. A
    circularity number alone would not distinguish that from a partly closed
    lid, so the angular location of the deficit is reported too.
    """
    m = iris_mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 60:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    circularity = area / max(math.pi * radius * radius, 1e-6)

    # deficit by angular sector, ignoring the top and bottom where the lids sit
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
    rr = np.hypot(xx - cx, yy - cy)
    inside = rr <= radius
    horizontal = ((ang < 35) | (ang > 325) | ((ang > 145) & (ang < 215)))
    band = inside & horizontal
    if band.sum() < 40:
        return None
    filled = (m.astype(bool) & band).sum() / band.sum()
    return {"circularity": float(circularity),
            "horizontal_fill": float(filled),
            "radius_px": float(radius)}
