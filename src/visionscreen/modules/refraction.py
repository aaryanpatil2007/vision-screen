"""Ballpark refractive error — a number to walk in with, not a prescription.

Every other module in this battery answers "is something wrong?". This one
answers "roughly how much, and in which direction?", which is the question
people actually have. It is also the easiest place in the whole system to lie
by accident, so the design is built around three rules:

1. **Fuse, don't pick.** Three independent signals bear on refraction —
   photorefraction (physics, signed), visual acuity (behavioural, unsigned),
   and the astigmatic dial (axis, weakly magnitude). Each is turned into a
   likelihood over the same grid and multiplied. A single signal quoted alone
   would either be over-confident (photorefraction on a noisy webcam) or
   ambiguous (acuity cannot tell myopia from hyperopia).

2. **Report the interval, not just the point.** The posterior's 95% credible
   interval is part of the result, not a footnote. "-2.25 D" is a claim this
   system cannot support; "-2.25 D, plausibly -1.50 to -3.00" is.

3. **Quantise to 0.25 D.** That is the step lenses are actually ground in.
   Reporting -2.13 D would imply a precision that does not exist anywhere in
   the chain.

The forward model is a published fit rather than one derived here:

    logMAR_unaided = logMAR_best_corrected + log10(1 + b^2)
    b^2 = M^2 + (C/2)^2                     (Thibos blur strength)

Blendowske 2015 fitted this across seven pooled datasets at R^2(adj) = 0.99.
Its own worked example -- 1 D of error costs 3 lines, not the "4 lines per
dioptre" of folklore -- is pinned as a test, so the citation cannot drift away
from the code. Using blur strength rather than |sphere| is what lets a
cylinder-only eye (M = 0, C = -2) correctly predict reduced acuity.

Four structural limits are encoded rather than hidden:

* **Acuity cannot see the sign of defocus.** b^2 is even in M, so +2 D and
  -2 D blur identically. Only photorefraction breaks the tie; without it the
  report says so rather than guessing.
* **Young hyperopes accommodate through their error.** A +2.00 D twenty-year-
  old reads 20/20 uncorrected. The likelihood subtracts available
  accommodation (Hofstetter minimum, A = 15 - 0.25*age) from positive defocus,
  which reproduces the measured blind spot: O'Donoghue 2012 found acuity
  screening detects myopia at 92% sensitivity but hyperopia at only 41-54%,
  with 11-15% of children with significant hyperopia or astigmatism reading
  0.20 logMAR or better.
* **Photorefraction saturates and then reverses past ~4 D** at a 5 mm pupil
  (Wu, Thibos & Candy 2018), so beyond that a strong error can imitate a
  milder one and no confident value is reported.
* **Nothing here can beat purpose-built hardware.** No published photoscreener
  achieves better than about +/-1.5 D 95% limits of agreement against
  cycloplegic refraction, and this has none of their optics. The credible
  interval is floored there, not at subjective refraction's own +/-0.78 D
  repeatability -- the device limit binds first and is roughly twice as wide.

What this module does *not* claim: there is no published continuous-dioptre
regression from a bare external eye photograph. The honest ceiling for imaging
with no added optics is binary myopia classification at AUC ~0.91-0.93 (Yang
2020), and every credible photorefractor uses controlled infrared illumination
at a known eccentricity -- which a webcam, with its infrared-cut filter and
autoexposure, does not have. The estimate here leans on the behavioural tests
for magnitude and uses the reflex only for sign and gross magnitude, which is
why its intervals are as wide as they are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- constants --

RAD_TO_ARCMIN = 180.0 * 60.0 / math.pi   # 3437.7; /1000 for mm*D -> 3.4377
BLUR_DISK_DEG_PER_MM_D = 0.057           # Strasburger 2018: b_deg = 0.057*p*D

DIOPTRE_STEP = 0.25        # lenses are ground in quarter-dioptre steps
SE_GRID = np.arange(-12.0, 8.0 + 1e-9, 0.125)
CYL_GRID = np.arange(0.0, 5.0 + 1e-9, 0.25)

# --- the acuity forward model ------------------------------------------------
# Blendowske, "Unaided Visual Acuity and Blur: A Simple Model", Optom Vis Sci
# 2015;92(6):e121-e125:
#
#     logMAR_unaided = logMAR_bestcorrected + log10(1 + b^2)
#
# fitted across seven pooled datasets with R^2(adj) = 0.99 and a regression
# standard error of 0.046 log units -- half that of Raasch's earlier quadratic
# in log b, which additionally diverges as b -> 0 and is undefined for an
# emmetrope. The model is *relative*: it predicts the drop from a person's own
# best-corrected acuity, not an absolute acuity.
BLENDOWSKE_FIT_SE_LOGMAR = 0.046   # on binned means; individuals scatter wider

# --- acuity measurement noise -----------------------------------------------
# Arditi & Cagenello 1993 put the 95% confidence on a letter-chart acuity at
# +/-0.10 logMAR and call that "the upper limit of reliability"; Siderov & Tiu
# 1999 measured 0.15 logMAR under real clinic conditions. A webcam at an
# uncontrolled distance is not better than a clinic.
SIGMA_ACUITY_LOGMAR = 0.15

# --- photorefraction ---------------------------------------------------------
# The dominant error is *calibration*, not fit residual. Bharadwaj et al. 2013
# ("Empirical variability in the calibration of slope-based eccentric
# photorefraction") found combined inter- and intra-subject calibration
# variability of about +/-40% of the mean slope. Worse, the constant is
# pigmentation-dependent: Sravani et al. 2015 found a Caucasian-derived
# calibration overestimated refraction in Indian eyes by 64 +/- 11%.
#
# So photorefractive uncertainty is *proportional* to the reading, not a fixed
# number of dioptres. A fixed sigma would be far too tight on a -6 D eye and
# needlessly loose near plano.
PHOTOREF_CALIBRATION_CV = 0.40     # Bharadwaj 2013
PHOTOREF_SIGMA_FLOOR_D = 0.50      # near plano, where the crescent vanishes

# Wu, Thibos & Candy 2018: the pupil luminance gradient is linear in refractive
# state only out to about 4 D at a 5 mm pupil, and beyond that "the gradient
# magnitude saturated and then reduced, leading to under-estimation". The
# reversal is the dangerous part -- a high myope can produce the same gradient
# as a moderate one -- so past this limit the reading is reported as a lower
# bound rather than a value.
PHOTOREF_LINEAR_LIMIT_D = 4.0      # at 5 mm; scales with pupil diameter
PHOTOREF_LINEAR_REF_PUPIL_MM = 5.0

# Roque et al. 2026 meta-analysis: non-cycloplegic *photoscreeners* -- the
# closest published analogue to this system -- underestimate refractive error
# by a pooled mean of -0.78 D, with a 95% prediction interval of -1.70 to
# +0.10 D. Without cycloplegia the eye can accommodate during the measurement,
# which reads as extra minus. This is a known bias, so it is corrected for
# rather than left in, and the prediction interval becomes its uncertainty.
NONCYCLO_BIAS_D = 0.78
NONCYCLO_BIAS_SIGMA_D = 0.46       # (1.70 + 0.10)/2 / 1.96

# --- the accuracy ceiling ----------------------------------------------------
# Two different floors apply, and conflating them was an error worth naming.
#
# The *reference standard* floor: subjective refraction agrees with itself to
# only about +/-0.78 D between clinicians (Bullimore 1998, n=86; MacKenzie 2008
# found a reproducibility coefficient of 0.78 D across 40 optometrists
# refracting a single eye) and +/-0.39 D for one clinician twice (Rosenfield &
# Chiu 1995). Nothing can be validated tighter than the thing it is measured
# against.
REFERENCE_STANDARD_CI_D = 0.78
#
# The *device* floor, which is what actually binds here and is roughly twice as
# wide. No published photoscreener achieves better than about +/-1.3 to 1.5 D
# 95% limits of agreement against cycloplegic refraction: plusoptiX +/-1.47 D
# (Mirzajani 2013), 2WIN -2.74 to +2.00 D (Kanclerz 2024), Retinomax -0.94 to
# +4.85 D with only 15.5% of readings within 0.5 D (Margines 2023, n=7,073).
# The pooled photoscreener prediction interval is -1.70 to +0.10 D (Roque 2025).
#
# This system has strictly worse optics than any of those devices -- no flash,
# no infrared, no controlled eccentricity, no fixed working distance. Reporting
# an interval narrower than the best purpose-built hardware would be claiming
# an accuracy the hardware cannot physically deliver, so the floor is set at
# the device figure, not the reference-standard one.
SCREENING_DEVICE_CI_FLOOR_D = 1.50

# Below about 0.75 D of cylinder most people report no meridian preference on a
# dial, so "no preference" is evidence of a small cylinder, not of none.
DIAL_DETECTION_THRESHOLD_D = 0.75

# Hofstetter's minimum amplitude of accommodation. Only about half is
# comfortably sustainable for a distance task.
def accommodation_reserve(age: float) -> float:
    """Dioptres of hyperopia a person of this age can comfortably mask."""
    amplitude = max(0.0, 15.0 - 0.25 * age)
    return 0.5 * amplitude


# --------------------------------------------------------------- the physics --

def blur_strength(se: float | np.ndarray, cyl: float | np.ndarray) -> np.ndarray:
    """Thibos blur strength B = sqrt(M^2 + J0^2 + J45^2) = sqrt(SE^2 + (C/2)^2).

    Cylinder is taken as a magnitude here; its axis does not change how much
    the eye is blurred, only in which direction.
    """
    return np.sqrt(np.asarray(se, float) ** 2 + (np.asarray(cyl, float) / 2.0) ** 2)


def blur_disk_arcmin(defocus_d: float, pupil_mm: float) -> float:
    """Geometric blur-disk diameter on the retina, in minutes of arc.

    Strasburger, Bach & Heinrich 2018: b(degrees) = 0.057 * p(mm) * D. Reported
    for context and for the pupil-dependence checks; the acuity prediction uses
    Blendowske's fit rather than this geometry directly, because the fit
    already absorbs the neural and optical factors that make threshold acuity
    better than the raw blur disk would suggest.
    """
    return abs(defocus_d) * pupil_mm * BLUR_DISK_DEG_PER_MM_D * 60.0


def expected_logmar(se, cyl, pupil_mm: float = 4.0, age: float | None = None,
                    logmar_best_corrected: float = 0.0):
    """Forward model: refractive error -> the acuity we expect to measure.

    Blendowske 2015, equation 4. Well-posed in this direction; the inverse is
    not, which is why the fusion below evaluates this over a grid instead of
    solving for `se`.

    Note what is *absent*: a pupil term. Blendowske's fit carries none, and the
    accompanying tutorial notes explicitly that "pupil size, although it
    influences acuity, does not appear in the equation" -- the fit was made on
    pupils restricted to 2-5 mm, which folds pupil variation into the residual
    scatter rather than modelling it. Pupil size is therefore handled here as
    an uncertainty term (see `_acuity_sigma`), not as a point correction. That
    is the honest reading of the source: inventing a pupil coefficient the
    fitted model does not have would be substituting my arithmetic for their
    data.
    """
    se = np.asarray(se, float)
    cyl = np.asarray(cyl, float)
    effective_se = se
    if age is not None:
        # hyperopia is partly self-corrected by accommodation; myopia never is
        reserve = accommodation_reserve(age)
        effective_se = np.where(se > 0, np.maximum(0.0, se - reserve), se)
    b2 = blur_strength(effective_se, cyl) ** 2
    return logmar_best_corrected + np.log10(1.0 + b2)


def _acuity_sigma(pupil_mm: float | None, cyl_present: bool,
                  tier_penalty: float = 1.0) -> float:
    """How far the acuity prediction can be off, all sources combined.

    Three terms, in rough order of size:

    * measurement noise on the acuity itself (0.15 logMAR in clinic conditions);
    * the forward model's own scatter -- 0.046 log units is the fit's standard
      error on *binned means*, and the paper is explicit that individual scatter
      is wider, so it is inflated here;
    * pupil size, which the model does not carry. Kamiya et al. 2012 measured
      the acuity cost of a fixed cylinder across 1-5 mm pupils and found the
      slope swing by roughly 3.5x, so an unmeasured or extreme pupil is a real
      and quantified source of error rather than a hedge.
    """
    var = SIGMA_ACUITY_LOGMAR ** 2 + (3.0 * BLENDOWSKE_FIT_SE_LOGMAR) ** 2
    if pupil_mm is None:
        var += 0.12 ** 2                     # unmeasured pupil
    elif not (2.0 <= pupil_mm <= 5.0):
        var += 0.10 ** 2                     # outside the range the fit covers
    if cyl_present:
        # Atchison & Mathur 2011 found astigmatic blur costs roughly twice what
        # spherical defocus of the same blur strength does, which Blendowske's
        # axis-independent b does not capture. The literature disagrees with
        # itself here, so the disagreement is carried as extra spread rather
        # than resolved by picking a side.
        var += 0.10 ** 2
    return math.sqrt(var) * tier_penalty


def photoref_sigma(reading_d: float, pupil_mm: float | None = None) -> float:
    """Uncertainty on a photorefractive reading, which scales with the reading.

    Calibration variability dominates and is multiplicative (Bharadwaj 2013,
    ~40% of the estimate), so a -6 D reading carries about 2.4 D of calibration
    uncertainty while a plano reading carries only the floor.
    """
    return math.hypot(PHOTOREF_CALIBRATION_CV * abs(reading_d),
                      PHOTOREF_SIGMA_FLOOR_D)


def photoref_linear_limit(pupil_mm: float | None) -> float:
    """Beyond this defocus the crescent gradient saturates and then reverses.

    Wu, Thibos & Candy 2018 measured the linear range as about 4 D at a 5 mm
    pupil. The limit scales with pupil diameter because a larger pupil samples
    more of the blurred wavefront before the gradient flattens.
    """
    p = pupil_mm if pupil_mm else PHOTOREF_LINEAR_REF_PUPIL_MM
    return PHOTOREF_LINEAR_LIMIT_D * (p / PHOTOREF_LINEAR_REF_PUPIL_MM)


def _quantise(d: float, step: float = DIOPTRE_STEP) -> float:
    return round(d / step) * step


# ----------------------------------------------------------------- the prior --

def _population_prior(se_grid: np.ndarray) -> np.ndarray:
    """Refractive error in the population is sharply peaked near plano with
    long tails, especially myopic.

    A Gaussian prior would drag a genuine -7 D myope toward -1 D. A Student-t
    (heavy-tailed) leaves the peak intact while refusing to rule out the tail,
    which is the behaviour we want: the prior should help when the data are
    weak and get out of the way when they are strong.
    """
    centre, scale, dof = -0.5, 1.25, 2.5
    z = (se_grid - centre) / scale
    return (1.0 + z ** 2 / dof) ** (-(dof + 1) / 2.0)


def _cylinder_prior(cyl_grid: np.ndarray) -> np.ndarray:
    """Most eyes have a little astigmatism; few have a lot. Half-normal."""
    return np.exp(-0.5 * (cyl_grid / 0.75) ** 2)


# ------------------------------------------------------------------- results --

@dataclass
class RefractionEstimate:
    """A ballpark refraction with its uncertainty attached, per eye."""

    eye: str
    spherical_equivalent: float | None
    se_interval: tuple[float, float] | None
    cylinder: float | None
    cyl_interval: tuple[float, float] | None
    axis: float | None
    sphere: float | None
    sign_known: bool
    confidence: str                       # indicative | broad | insufficient
    sources: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def prescription_string(self) -> str:
        if self.spherical_equivalent is None or self.sphere is None:
            return "not estimable"
        if self.cylinder and self.cylinder >= 0.25 and self.axis is not None:
            return (f"{self.sphere:+.2f} / {-abs(self.cylinder):.2f} "
                    f"x {self.axis:.0f}")
        return f"{self.sphere:+.2f} DS"

    @property
    def plain_summary(self) -> str:
        if self.spherical_equivalent is None:
            return "This screening could not put a number on your focus."
        lo, hi = self.se_interval
        se = self.spherical_equivalent
        if not self.sign_known:
            return (f"Your eye is out of focus by roughly {abs(se):.2f} dioptres, "
                    f"but this test could not tell whether that is short sight or "
                    f"long sight. An eye exam can, in about a minute.")
        kind = ("short sight (myopia)" if se < -0.5 else
                "long sight (hyperopia)" if se > 0.5 else
                "close to no focusing error")
        if abs(se) <= 0.5:
            return (f"Your focus looks {kind} — within about "
                    f"{max(abs(lo), abs(hi)):.2f} dioptres of neutral.")
        return (f"Roughly {abs(se):.2f} dioptres of {kind}. "
                f"A real exam would most likely land somewhere between "
                f"{lo:+.2f} and {hi:+.2f}.")


# ------------------------------------------------------------------- fusion --

def estimate_refraction(
    eye: str = "unspecified",
    *,
    photoref_sphere: float | None = None,
    photoref_cyl: float | None = None,
    photoref_axis: float | None = None,
    photoref_tier: str = "inconclusive",
    acuity_logmar: float | None = None,
    acuity_tier: str = "inconclusive",
    dial_axis: float | None = None,
    dial_detected: bool | None = None,
    pupil_mm: float | None = None,
    age: float | None = None,
) -> RefractionEstimate:
    """Combine every signal that bears on refraction into one posterior.

    Each argument is optional; the estimate degrades gracefully as signals drop
    out, and reports which ones it actually used. `dial_detected=False` means
    the person saw the dial as uniform, which is informative (small cylinder),
    as distinct from `None`, which means the test was not done or failed.
    """
    sources: list[str] = []
    caveats: list[str] = []

    if pupil_mm is None:
        caveats.append(
            "Pupil size was not measured; 4 mm was assumed. Pupil size scales "
            "how much a given focusing error blurs vision, so a real pupil "
            "measurement would tighten this estimate."
        )

    se = SE_GRID[:, None]                 # (S, 1)
    cyl = CYL_GRID[None, :]               # (1, C)
    log_post = np.log(_population_prior(SE_GRID))[:, None] \
        + np.log(_cylinder_prior(CYL_GRID))[None, :]

    # --- photorefraction: the only signed signal ---------------------------
    sign_known = False
    saturated = False
    if photoref_sphere is not None and photoref_tier in ("measured", "weak-signal"):
        # photoref reports sphere; convert to spherical equivalent, SE = S + C/2
        pr_cyl = abs(photoref_cyl or 0.0)
        pr_se_raw = photoref_sphere + (-pr_cyl) / 2.0

        # Correct the known non-cycloplegic bias rather than carrying it as an
        # unmodelled error: without cycloplegia the eye accommodates during the
        # measurement and reads too minus, by a pooled -0.78 D for photoscreeners.
        pr_se = pr_se_raw + NONCYCLO_BIAS_D
        sigma = math.hypot(photoref_sigma(pr_se_raw, pupil_mm), NONCYCLO_BIAS_SIGMA_D)
        if photoref_tier != "measured":
            sigma *= 2.0
        log_post = log_post - 0.5 * ((se - pr_se) / sigma) ** 2
        sign_known = photoref_tier == "measured"
        sources.append(f"photorefraction ({photoref_tier})")
        caveats.append(
            "The red-reflex reading was adjusted by +{:.2f} D for the fact that "
            "the eye can focus during the measurement, which makes it read too "
            "short-sighted. Eye clinics use drops to stop that; this cannot."
            .format(NONCYCLO_BIAS_D))

        limit = photoref_linear_limit(pupil_mm)
        if abs(pr_se_raw) > limit:
            saturated = True
            caveats.append(
                f"The reading was beyond the {limit:.1f} D range where this "
                "method stays proportional. Past that point the signal flattens "
                "and then runs backwards, so a strong error can imitate a milder "
                "one — treat the number as 'at least this much', not as a value."
            )

        if photoref_cyl is not None:
            sc = photoref_sigma(pr_cyl, pupil_mm)
            if photoref_tier != "measured":
                sc *= 2.0
            log_post = log_post - 0.5 * ((cyl - pr_cyl) / sc) ** 2
        if photoref_tier != "measured":
            caveats.append(
                "The red-reflex measurement was weak, so the direction of the "
                "error (short vs long sight) is not certain."
            )

    # --- acuity: strong on magnitude, blind to sign ------------------------
    if acuity_logmar is not None and acuity_tier in ("measured", "weak-signal"):
        predicted = expected_logmar(se, cyl, pupil_mm=pupil_mm or 4.0, age=age)
        sigma = _acuity_sigma(pupil_mm, cyl_present=True,
                              tier_penalty=1.0 if acuity_tier == "measured" else 1.8)
        log_post = log_post - 0.5 * ((acuity_logmar - predicted) / sigma) ** 2
        sources.append(f"visual acuity ({acuity_tier})")
        if not sign_known:
            caveats.append(
                "Acuity measures how blurred vision is, not which way. Without "
                "a usable red-reflex reading this estimate cannot separate "
                "short sight from long sight."
            )
        if age is None:
            caveats.append(
                "Age was not given. Under about 40 the eye can focus through "
                "long sight, so long sight can hide from an acuity test; "
                "supplying age would let the estimate account for that."
            )

    # --- astigmatic dial: axis, and a soft bound on magnitude --------------
    axis = None
    if dial_detected is True:
        # a reported meridian preference means the cylinder is above the
        # threshold where a dial becomes visible at all
        log_post = log_post + np.log(
            1.0 / (1.0 + np.exp(-(cyl - DIAL_DETECTION_THRESHOLD_D) / 0.25)) + 1e-9)
        axis = dial_axis
        sources.append("astigmatic dial")
    elif dial_detected is False:
        log_post = log_post + np.log(
            1.0 / (1.0 + np.exp((cyl - DIAL_DETECTION_THRESHOLD_D) / 0.25)) + 1e-9)
        sources.append("astigmatic dial (no meridian preference)")
    if axis is None and photoref_axis is not None:
        axis = photoref_axis

    if not sources:
        return RefractionEstimate(
            eye=eye, spherical_equivalent=None, se_interval=None, cylinder=None,
            cyl_interval=None, axis=None, sphere=None, sign_known=False,
            confidence="insufficient",
            caveats=["No test in this screening produced a usable refraction signal."],
        )

    # --- posterior ---------------------------------------------------------
    post = np.exp(log_post - log_post.max())
    total = post.sum()
    if not np.isfinite(total) or total <= 0:
        return RefractionEstimate(
            eye=eye, spherical_equivalent=None, se_interval=None, cylinder=None,
            cyl_interval=None, axis=None, sphere=None, sign_known=False,
            confidence="insufficient", sources=sources,
            caveats=["The signals disagreed too strongly to combine."],
        )
    post /= total

    se_marg = post.sum(axis=1)
    cyl_marg = post.sum(axis=0)
    se_mean = float((SE_GRID * se_marg).sum())
    cyl_mean = float((CYL_GRID * cyl_marg).sum())
    se_lo, se_hi = _credible_interval(SE_GRID, se_marg)
    cyl_lo, cyl_hi = _credible_interval(CYL_GRID, cyl_marg)

    # A tighter interval than subjective refraction's own between-clinician
    # agreement would be a claim to beat the reference standard. Widen to the
    # floor rather than report a precision nothing in this chain can support.
    if (se_hi - se_lo) < SCREENING_DEVICE_CI_FLOOR_D:
        mid = 0.5 * (se_hi + se_lo)
        se_lo = mid - SCREENING_DEVICE_CI_FLOOR_D / 2.0
        se_hi = mid + SCREENING_DEVICE_CI_FLOOR_D / 2.0
        caveats.append(
            "The range shown is held at {:.2f} D wide. No purpose-built vision "
            "screener does better than that against a proper eye exam, and this "
            "has none of their hardware, so a narrower range would be claiming "
            "an accuracy the equipment cannot deliver."
            .format(SCREENING_DEVICE_CI_FLOOR_D))

    width = se_hi - se_lo
    if saturated:
        confidence = "insufficient"
    elif width <= SCREENING_DEVICE_CI_FLOOR_D + 0.3 and sign_known:
        confidence = "indicative"
    elif width <= 3.0:
        confidence = "broad"
    else:
        confidence = "insufficient"

    cyl_q = _quantise(cyl_mean)
    if cyl_q < 0.25:
        cyl_q, axis = 0.0, None
    se_q = _quantise(se_mean)
    # sphere = SE - C/2, with C carried in minus-cylinder convention
    sphere_q = _quantise(se_q + cyl_q / 2.0)

    if confidence == "insufficient":
        caveats.append(
            "The range this screening can support is too wide to be worth "
            "quoting a single number from."
        )

    return RefractionEstimate(
        eye=eye,
        spherical_equivalent=se_q,
        se_interval=(_quantise(se_lo), _quantise(se_hi)),
        cylinder=cyl_q,
        cyl_interval=(_quantise(cyl_lo), _quantise(cyl_hi)),
        axis=(round(axis, 0) if axis is not None else None),
        sphere=sphere_q,
        sign_known=sign_known,
        confidence=confidence,
        sources=sources,
        caveats=caveats,
    )


def _credible_interval(grid: np.ndarray, marginal: np.ndarray,
                       mass: float = 0.95) -> tuple[float, float]:
    """Equal-tailed interval containing `mass` of the posterior."""
    cdf = np.cumsum(marginal)
    cdf = cdf / cdf[-1]
    tail = (1.0 - mass) / 2.0
    lo = float(grid[np.searchsorted(cdf, tail)])
    hi = float(grid[min(np.searchsorted(cdf, 1.0 - tail), len(grid) - 1)])
    return lo, hi
