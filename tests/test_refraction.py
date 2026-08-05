"""The refraction estimate must be honest before it is precise.

The forward model is checked against the clinical rule-of-thumb table relating
uncorrected myopia to distance acuity, because that table is the thing a
clinician would sanity-check the output against. The fusion is then checked for
the properties that keep the number defensible: it must widen when signals are
weak, refuse to guess a sign it cannot see, and never report finer than 0.25 D.
"""
from __future__ import annotations

import numpy as np
import pytest

from visionscreen.modules.refraction import (
    blur_disk_arcmin,
    DIOPTRE_STEP,
    accommodation_reserve,
    blur_strength,
    estimate_refraction,
    expected_logmar,
)

# Uncorrected myopia -> distance acuity, the range every clinical text gives.
# (spherical equivalent, logMAR low, logMAR high) at a normal ~4 mm pupil.
CLINICAL_TABLE = [
    (-0.50, 0.05, 0.35),      # 20/22 - 20/45
    (-1.00, 0.30, 0.60),      # 20/40 - 20/80
    (-2.00, 0.60, 1.00),      # 20/80 - 20/200
    (-3.00, 0.80, 1.20),      # 20/125 - 20/320
    (-4.00, 1.00, 1.40),      # 20/200 - 20/500
]


@pytest.mark.parametrize("se,lo,hi", CLINICAL_TABLE)
def test_forward_model_matches_clinical_acuity_table(se, lo, hi):
    """A model that cannot reproduce known myopia-to-acuity behaviour cannot be
    inverted into a meaningful prescription."""
    got = float(expected_logmar(se, 0.0, pupil_mm=4.0))
    assert lo <= got <= hi, f"{se:+.2f} D predicted logMAR {got:.2f}, expected {lo}-{hi}"


def test_emmetropia_predicts_normal_acuity():
    assert float(expected_logmar(0.0, 0.0, pupil_mm=4.0)) == pytest.approx(0.0, abs=0.02)


def test_blur_disk_geometry_scales_with_pupil():
    """b(deg) = 0.057 * p(mm) * D — Strasburger 2018. The geometry is linear in
    both terms even though the fitted acuity model does not carry a pupil."""
    assert blur_disk_arcmin(1.0, 4.0) == pytest.approx(0.057 * 4.0 * 60.0)
    assert blur_disk_arcmin(2.0, 4.0) == pytest.approx(2 * blur_disk_arcmin(1.0, 4.0))
    assert blur_disk_arcmin(1.0, 6.0) > blur_disk_arcmin(1.0, 3.0)


def test_pupil_enters_as_uncertainty_not_as_a_point_correction():
    """Blendowske's fit carries no pupil term — it was made on 2-5 mm pupils,
    which folds pupil variation into residual scatter. Inventing a coefficient
    the published model does not have would be substituting arithmetic for
    their data, so pupil widens the error bar instead."""
    from visionscreen.modules.refraction import _acuity_sigma
    assert float(expected_logmar(-1.0, 0.0, pupil_mm=2.5)) == \
           pytest.approx(float(expected_logmar(-1.0, 0.0, pupil_mm=6.0)))
    assert _acuity_sigma(None, False) > _acuity_sigma(4.0, False)
    assert _acuity_sigma(7.0, False) > _acuity_sigma(4.0, False)


def test_forward_model_reproduces_the_papers_own_worked_example():
    """Blendowske states the model gives "a drop-off of 3 lines (0.3 log units)
    at a refractive error of 1 D", explicitly contrasting it with the '4 lines
    per dioptre' rule of thumb. If this drifts, the citation is no longer honest."""
    assert float(expected_logmar(-1.0, 0.0)) == pytest.approx(0.301, abs=0.005)


def test_model_is_defined_and_flat_at_plano():
    """Raasch's earlier log-b quadratic diverges as b -> 0 and is undefined for
    an emmetrope; this one is well-behaved, and its slope at zero is zero
    because blur is quadratic near best focus, not linear."""
    assert float(expected_logmar(0.0, 0.0)) == pytest.approx(0.0)
    lo = float(expected_logmar(-0.05, 0.0))
    assert lo < 0.005          # essentially flat, not divergent


def test_model_is_relative_to_best_corrected_acuity():
    """It predicts the drop from a person's own baseline, not an absolute
    acuity — someone with amblyopia starts lower and stays lower."""
    assert float(expected_logmar(-1.0, 0.0, logmar_best_corrected=0.3)) == \
           pytest.approx(0.3 + 0.301, abs=0.005)


def test_blur_strength_counts_cylinder_at_half_weight():
    """Thibos power vectors: a 2 D cylinder blurs like a 1 D sphere."""
    assert blur_strength(0.0, 2.0) == pytest.approx(1.0)
    assert blur_strength(-1.0, 0.0) == pytest.approx(1.0)
    assert blur_strength(-3.0, 4.0) == pytest.approx(np.hypot(3.0, 2.0))


def test_cylinder_alone_reduces_acuity():
    """An eye with no spherical error but 2 D of cylinder does not see 20/20."""
    assert float(expected_logmar(0.0, 2.0, pupil_mm=4.0)) > 0.3


def test_young_hyperope_accommodates_through_the_error():
    """+2.00 D at twenty reads the chart fine; at seventy it does not. If the
    model missed this it would confidently report emmetropia for every young
    hyperope — the exact population screening is meant to catch."""
    young = float(expected_logmar(2.0, 0.0, pupil_mm=4.0, age=20))
    old = float(expected_logmar(2.0, 0.0, pupil_mm=4.0, age=70))
    assert young == pytest.approx(0.0, abs=0.02)
    assert old > 0.5


def test_myopia_is_never_accommodated_away():
    """Accommodation only adds plus power, so it cannot help a myope."""
    for age in (10, 25, 45, 80):
        assert float(expected_logmar(-2.0, 0.0, age=age)) > 0.5


def test_accommodation_reserve_declines_with_age():
    assert accommodation_reserve(10) > accommodation_reserve(40)
    assert accommodation_reserve(70) == 0.0


# ------------------------------------------------------------------ fusion --

def test_photorefraction_and_acuity_agree_tightens_the_interval():
    """Two signals pointing at the same answer should be worth more than one."""
    one = estimate_refraction(
        acuity_logmar=0.75, acuity_tier="measured", pupil_mm=4.0, age=25)
    both = estimate_refraction(
        photoref_sphere=-2.0, photoref_cyl=0.0, photoref_tier="measured",
        acuity_logmar=0.75, acuity_tier="measured", pupil_mm=4.0, age=25)
    w_one = one.se_interval[1] - one.se_interval[0]
    w_both = both.se_interval[1] - both.se_interval[0]
    assert w_both < w_one
    assert both.spherical_equivalent == pytest.approx(-2.0, abs=0.6)


def test_acuity_alone_cannot_name_the_sign():
    """Blur is blur. Reporting "myopia" from acuity alone would be a guess."""
    est = estimate_refraction(acuity_logmar=0.8, acuity_tier="measured",
                              pupil_mm=4.0, age=30)
    assert est.sign_known is False
    assert "could not tell whether" in est.plain_summary
    assert any("which way" in c for c in est.caveats)


def test_photorefraction_supplies_the_sign():
    est = estimate_refraction(
        photoref_sphere=-2.5, photoref_cyl=0.5, photoref_tier="measured",
        acuity_logmar=0.9, acuity_tier="measured", pupil_mm=4.5, age=30)
    assert est.sign_known is True
    assert est.spherical_equivalent < 0
    assert "short sight" in est.plain_summary


def test_weak_signals_widen_rather_than_lie():
    strong = estimate_refraction(
        photoref_sphere=-2.0, photoref_tier="measured",
        acuity_logmar=0.75, acuity_tier="measured", pupil_mm=4.0, age=25)
    weak = estimate_refraction(
        photoref_sphere=-2.0, photoref_tier="weak-signal",
        acuity_logmar=0.75, acuity_tier="weak-signal", pupil_mm=4.0, age=25)
    assert (weak.se_interval[1] - weak.se_interval[0]) > \
           (strong.se_interval[1] - strong.se_interval[0])
    assert weak.confidence in ("broad", "insufficient")


def test_no_signals_returns_not_estimable():
    est = estimate_refraction()
    assert est.spherical_equivalent is None
    assert est.confidence == "insufficient"
    assert est.prescription_string == "not estimable"


def test_output_is_quantised_to_quarter_dioptres():
    """Lenses come in 0.25 D steps; anything finer is invented precision."""
    est = estimate_refraction(
        photoref_sphere=-2.13, photoref_cyl=0.87, photoref_tier="measured",
        acuity_logmar=0.71, acuity_tier="measured", pupil_mm=4.3, age=31)
    for value in (est.spherical_equivalent, est.sphere, est.cylinder,
                  *est.se_interval):
        assert abs(value / DIOPTRE_STEP - round(value / DIOPTRE_STEP)) < 1e-9, value


def test_dial_preference_pushes_cylinder_up_and_absence_pushes_it_down():
    seen = estimate_refraction(acuity_logmar=0.5, acuity_tier="measured",
                               dial_detected=True, dial_axis=90.0, age=30)
    unseen = estimate_refraction(acuity_logmar=0.5, acuity_tier="measured",
                                 dial_detected=False, age=30)
    assert seen.cylinder > unseen.cylinder
    assert seen.axis == 90


def test_prescription_string_uses_minus_cylinder_convention():
    est = estimate_refraction(
        photoref_sphere=-1.5, photoref_cyl=1.0, photoref_axis=180.0,
        photoref_tier="measured", acuity_logmar=0.6, acuity_tier="measured",
        dial_detected=True, dial_axis=180.0, pupil_mm=4.0, age=30)
    s = est.prescription_string
    assert "/" in s and "x" in s
    assert "-" in s.split("/")[1]          # cylinder must be negative


def test_emmetrope_is_reported_as_near_neutral_not_as_a_prescription():
    est = estimate_refraction(
        photoref_sphere=0.0, photoref_cyl=0.0, photoref_tier="measured",
        acuity_logmar=0.0, acuity_tier="measured", dial_detected=False,
        pupil_mm=4.0, age=25)
    assert abs(est.spherical_equivalent) <= 0.5
    assert "no focusing error" in est.plain_summary or "neutral" in est.plain_summary


def test_moderate_myopia_is_not_shrunk_toward_zero_by_the_prior():
    """A heavy-tailed prior must let strong evidence win; a Gaussian prior
    would drag a real reading up toward the population mean."""
    est = estimate_refraction(
        photoref_sphere=-3.0, photoref_tier="measured",
        acuity_logmar=1.0, acuity_tier="measured", pupil_mm=4.0, age=25)
    assert est.spherical_equivalent < -1.75


def test_reading_beyond_the_linear_range_is_not_reported_as_a_value():
    """Wu, Thibos & Candy 2018: past ~4 D at a 5 mm pupil the crescent gradient
    saturates and then *reverses*, so a high myope can imitate a moderate one.
    Reporting a confident number there would be reporting a number the method
    cannot distinguish from a different one."""
    est = estimate_refraction(
        photoref_sphere=-7.0, photoref_tier="measured",
        acuity_logmar=1.3, acuity_tier="measured", pupil_mm=4.0, age=25)
    assert est.confidence == "insufficient"
    assert any("runs backwards" in c or "at least this much" in c
               for c in est.caveats), est.caveats


def test_photorefraction_uncertainty_scales_with_the_reading():
    """Calibration variability is ~40% of the estimate (Bharadwaj 2013), so it
    is multiplicative. A fixed sigma would be far too tight on a -6 D eye."""
    from visionscreen.modules.refraction import photoref_sigma
    assert photoref_sigma(0.0) == pytest.approx(0.50, abs=0.01)
    assert photoref_sigma(-6.0) > 2.0
    assert photoref_sigma(-6.0) > 3 * photoref_sigma(0.0) * 0.7


def test_non_cycloplegic_bias_is_corrected_toward_plus():
    """Without drops the eye accommodates during the measurement and reads too
    minus — a pooled -0.78 D for photoscreeners (Roque 2026). Leaving it in
    would systematically over-call myopia."""
    est = estimate_refraction(
        photoref_sphere=-2.0, photoref_tier="measured", pupil_mm=4.0, age=25)
    assert est.spherical_equivalent > -2.0
    assert any("focus during the measurement" in c for c in est.caveats)


def test_interval_never_claims_to_beat_the_reference_standard():
    """Subjective refraction agrees with itself to about +/-0.78 D between
    clinicians (Bullimore 1998, MacKenzie 2008). An interval tighter than that
    would be claiming more precision than the test being compared against."""
    from visionscreen.modules.refraction import SCREENING_DEVICE_CI_FLOOR_D as REFERENCE_STANDARD_CI_D
    est = estimate_refraction(
        photoref_sphere=-1.0, photoref_tier="measured",
        acuity_logmar=0.30, acuity_tier="measured",
        dial_detected=False, pupil_mm=4.0, age=25)
    lo, hi = est.se_interval
    assert (hi - lo) >= REFERENCE_STANDARD_CI_D - 1e-9, (lo, hi)


def test_conflicting_signals_do_not_produce_a_confident_answer():
    """Photorefraction saying plano while acuity says heavily blurred is a
    contradiction; the result must not average them into a smug midpoint."""
    est = estimate_refraction(
        photoref_sphere=0.0, photoref_tier="measured",
        acuity_logmar=1.2, acuity_tier="measured", pupil_mm=4.0, age=25)
    assert est.confidence != "indicative"
