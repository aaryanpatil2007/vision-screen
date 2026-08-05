"""Anterior-segment findings, checked against eyes whose truth we constructed.

Each test renders an eye with one property deliberately set — a dim reflex, a
yellow sclera, a low lid — and asserts both that the property is detected and,
just as importantly, that a normal eye is *not* flagged. A screening test that
finds everything is worthless, and the false-positive direction is the one that
sends people to appointments they did not need.

The white-balance tests exist because colour is the weakest link here: they pin
the rule that an uncalibrated frame never produces an actionable colour finding.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from visionscreen.modules.anterior import (
    HVID_MM,
    measure_arcus,
    measure_iris_boundary,
    measure_mrd1,
    measure_red_reflex,
    measure_sclera_colour,
    px_per_mm_from_iris,
    score_arcus,
    score_ptosis,
    score_red_reflex,
    score_sclera,
    white_balance,
)

SIZE = (120, 160)          # h, w — a typical eye crop
CENTER = (80.0, 60.0)      # x, y
IRIS_R = 34.0
PUPIL_R = 12.0


def _disk(shape, center, radius) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(xx - center[0], yy - center[1]) <= radius


def _masks(iris_r=IRIS_R, pupil_r=PUPIL_R):
    iris = _disk(SIZE, CENTER, iris_r) & ~_disk(SIZE, CENTER, pupil_r)
    pupil = _disk(SIZE, CENTER, pupil_r)
    sclera = np.zeros(SIZE, bool)
    sclera[40:80, 10:150] = True
    sclera &= ~_disk(SIZE, CENTER, iris_r + 1)
    return sclera, iris, pupil


def _eye(reflex=(190, 70, 60), sclera_rgb=(238, 236, 232), iris_rgb=(95, 80, 70),
         reflex_patch=None, arcus=False) -> np.ndarray:
    """A crude but sufficient colour eye: sclera, iris, pupil filled with a reflex."""
    sclera, iris, pupil = _masks()
    img = np.full((*SIZE, 3), 40, np.uint8)
    img[sclera] = sclera_rgb
    img[iris] = iris_rgb
    img[pupil] = reflex
    if reflex_patch is not None:
        yy, xx = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        shadow = (np.hypot(xx - (CENTER[0] + 5), yy - CENTER[1]) < 6) & pupil
        img[shadow] = reflex_patch
    if arcus:
        yy, xx = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        r = np.hypot(xx - CENTER[0], yy - CENTER[1])
        ring = (r > IRIS_R * 0.88) & (r <= IRIS_R * 0.98)
        img[ring] = (205, 200, 195)
    return img


# ------------------------------------------------------------- calibration --

def test_iris_width_gives_a_millimetre_scale():
    _, iris, _ = _masks()
    scale = px_per_mm_from_iris(iris)
    assert scale == pytest.approx(2 * IRIS_R / HVID_MM, rel=0.05)


def test_scale_is_none_when_the_iris_is_not_visible():
    assert px_per_mm_from_iris(np.zeros(SIZE, bool)) is None


def test_white_balance_neutralises_a_colour_cast():
    warm = np.full((10, 10, 3), (220, 190, 150), np.uint8)
    out, ok = white_balance(warm, (220, 190, 150))
    assert ok
    assert out[..., 0].mean() == pytest.approx(out[..., 2].mean(), abs=3)


def test_white_balance_refuses_a_clipped_reference():
    """A blown-out white patch carries no colour information — using it would
    silently apply a wrong correction."""
    img = np.full((10, 10, 3), (200, 180, 150), np.uint8)
    _, ok = white_balance(img, (255, 255, 255))
    assert ok is False
    _, ok2 = white_balance(img, None)
    assert ok2 is False


# -------------------------------------------------------------- red reflex --

def test_healthy_reflex_is_not_flagged():
    _, _, pupil = _masks()
    m = measure_red_reflex(_eye(), pupil)
    f = score_red_reflex(m, m)
    assert "flags" not in f.metrics


def test_reflex_can_never_reach_an_actionable_tier():
    """A laptop has no flash. CRADLE — a purpose-built leukocoria app using a
    real camera flash — reported 90% sensitivity from its developers and 15.4%
    in independent prospective validation. A test that looks like a cataract
    detector while missing six in seven cases is worse than no test, because it
    converts "I should get this checked" into false reassurance."""
    _, _, pupil = _masks()
    good = measure_red_reflex(_eye(), pupil)
    dim = measure_red_reflex(_eye(reflex=(60, 25, 20)), pupil)
    pale = measure_red_reflex(_eye(reflex=(215, 213, 210)), pupil)
    for a, b in ((good, good), (good, dim), (pale, None), (dim, dim)):
        f = score_red_reflex(a, b)
        assert f.tier != "measured", f.summary


def test_clear_reflex_is_not_presented_as_an_all_clear():
    """The false-negative direction is the dangerous one here."""
    _, _, pupil = _masks()
    m = measure_red_reflex(_eye(), pupil)
    f = score_red_reflex(m, m)
    assert "cannot rule anything out" in f.summary
    assert "not as an all-clear" in f.summary


def test_asymmetric_reflex_is_flagged():
    """The Brueckner signal: one eye returning much less light than the other."""
    _, _, pupil = _masks()
    bright = measure_red_reflex(_eye(reflex=(200, 80, 70)), pupil)
    dim = measure_red_reflex(_eye(reflex=(70, 28, 24)), pupil)
    f = score_red_reflex(bright, dim)
    assert "asymmetric reflex" in f.metrics["flags"]
    assert f.tier == "weak-signal"          # never promoted to a confident claim


def test_patchy_reflex_is_flagged():
    _, _, pupil = _masks()
    even = measure_red_reflex(_eye(), pupil)
    patchy = measure_red_reflex(_eye(reflex_patch=(20, 10, 10)), pupil)
    assert patchy["cv"] > even["cv"]
    f = score_red_reflex(patchy, patchy)
    assert any("patchy" in x for x in f.metrics["flags"])


def test_pale_reflex_is_flagged_even_from_one_eye():
    """Leukocoria can be an emergency, so it must not require the other eye."""
    _, _, pupil = _masks()
    pale = measure_red_reflex(_eye(reflex=(215, 213, 210)), pupil)
    f = score_red_reflex(pale, None)
    assert any("pale reflex" in x for x in f.metrics["flags"])


def test_reflex_never_names_a_diagnosis():
    """The summary must describe the observation, not assert the disease."""
    _, _, pupil = _masks()
    dim = measure_red_reflex(_eye(reflex=(60, 25, 20)), pupil)
    f = score_red_reflex(measure_red_reflex(_eye(), pupil), dim)
    lowered = f.summary.lower()
    for word in ("cataract", "retinoblastoma", "tumour", "tumor", "diagnosis of"):
        assert word not in lowered, f"summary asserts '{word}'"
    assert "not a diagnosis" in lowered


def test_reflex_measurement_excludes_the_corneal_glint():
    """An unmasked specular highlight is far brighter than the reflex and would
    dominate the statistic — the same failure that broke reflex localisation."""
    _, _, pupil = _masks()
    plain = _eye()
    glinted = plain.copy()
    glinted[int(CENTER[1]) - 2:int(CENTER[1]) + 2,
            int(CENTER[0]) - 2:int(CENTER[0]) + 2] = 255
    a = measure_red_reflex(plain, pupil)
    b = measure_red_reflex(glinted, pupil)
    assert b["mean"] == pytest.approx(a["mean"], rel=0.08)


def test_missing_reflex_is_inconclusive_not_negative():
    f = score_red_reflex(None, None)
    assert f.tier == "inconclusive"
    assert f.retakes


# ------------------------------------------------------------------ ptosis --

def test_mrd1_converts_pixels_to_millimetres():
    px_per_mm = 2 * IRIS_R / HVID_MM
    assert measure_mrd1(50.0, 60.0, px_per_mm) == pytest.approx(10.0 / px_per_mm)


def test_normal_lids_are_not_flagged():
    f = score_ptosis(4.2, 4.4)
    assert f.tier == "measured"
    assert "flags" not in f.metrics


def test_low_lid_is_flagged():
    f = score_ptosis(1.4, 4.3)
    assert any("low lid" in x for x in f.metrics["flags"])


def test_asymmetric_lids_are_flagged_even_when_both_are_in_range():
    f = score_ptosis(2.3, 4.6)
    assert "asymmetric lids" in f.metrics["flags"]


def test_lid_position_is_refused_when_gaze_is_off_axis():
    """The lid follows the eye, so a glance up or down fakes the measurement."""
    f = score_ptosis(1.0, 4.5, gaze_ok=False)
    assert f.tier == "inconclusive"
    assert "flags" not in f.metrics


# ------------------------------------------------------------ sclera colour --

def test_normal_sclera_with_reference_is_clear():
    sclera, _, _ = _masks()
    m = measure_sclera_colour(_eye(), sclera)
    f = score_sclera(m, m, calibrated=True)
    assert "flags" not in f.metrics


def test_colour_findings_stay_below_actionable_while_thresholds_are_unvalidated():
    """A white reference fixes the camera, not the cut-points. The b* and
    redness thresholds were chosen by eye with no labelled corpus behind them,
    so no colour reading may reach the tier the report treats as a result —
    even on a perfectly white-balanced frame."""
    from visionscreen.modules.anterior import SCLERA_COLOUR_CALIBRATED
    sclera, _, _ = _masks()
    for img in (_eye(), _eye(sclera_rgb=(240, 220, 120))):
        f = score_sclera(measure_sclera_colour(img, sclera),
                         measure_sclera_colour(img, sclera), calibrated=True)
        if not SCLERA_COLOUR_CALIBRATED:
            assert f.tier != "measured", f.summary
            assert "not yet been set" in f.summary


def test_yellow_sclera_is_flagged_when_calibrated():
    sclera, _, _ = _masks()
    m = measure_sclera_colour(_eye(sclera_rgb=(240, 220, 120)), sclera)
    assert m["b_star"] > 18.0
    f = score_sclera(m, m, calibrated=True)
    assert any("yellow" in x for x in f.metrics["flags"])
    assert "liver" in f.summary.lower()
    assert f.tier in ("measured", "weak-signal")


def test_red_sclera_is_flagged_when_calibrated():
    sclera, _, _ = _masks()
    m = measure_sclera_colour(_eye(sclera_rgb=(235, 90, 90)), sclera)
    f = score_sclera(m, m, calibrated=True)
    assert any("redness" in x for x in f.metrics["flags"])


def test_uncalibrated_colour_is_never_actionable():
    """This is the whole point of the white-paper step: without a reference the
    camera cannot separate a yellow eye from yellow light, so no colour claim
    may reach a tier a user would act on."""
    sclera, _, _ = _masks()
    m = measure_sclera_colour(_eye(sclera_rgb=(240, 220, 120)), sclera)
    f = score_sclera(m, m, calibrated=False)
    assert f.tier in ("weak-signal", "inconclusive")
    assert "liver" not in f.summary.lower()
    assert "white" in f.summary.lower() or any("white" in r.lower() for r in f.retakes)


def test_sclera_measurement_erodes_away_the_boundary():
    """Boundary pixels blend sclera with iris and lashes, and they bias both
    statistics toward the alarming direction."""
    sclera, _, _ = _masks()
    img = _eye()
    m = measure_sclera_colour(img, sclera)
    assert m["pixels"] < sclera.sum()
    assert m["b_star"] == pytest.approx(0.0, abs=6.0)


# ------------------------------------------------------------------- arcus --

def test_arcus_ring_is_detected():
    _, iris, _ = _masks()
    gray = cv2.cvtColor(_eye(arcus=True), cv2.COLOR_RGB2GRAY)
    c = measure_arcus(gray, iris, CENTER, IRIS_R)
    assert c is not None and c > 0.18


def test_plain_iris_shows_no_arcus():
    _, iris, _ = _masks()
    gray = cv2.cvtColor(_eye(), cv2.COLOR_RGB2GRAY)
    c = measure_arcus(gray, iris, CENTER, IRIS_R)
    assert c is not None and c < 0.18
    assert score_arcus(c, c).tier == "measured"
    assert "flags" not in score_arcus(c, c).metrics


def test_arcus_advice_depends_on_age():
    """Common and harmless at 70, worth a cholesterol check at 30 — the same
    measurement means different things, so age drives the wording."""
    young = score_arcus(0.30, 0.30, age=30)
    old = score_arcus(0.30, 0.30, age=70)
    assert "cholesterol" in young.summary.lower()
    assert "needs nothing done" in old.summary.lower()
    assert young.tier == "weak-signal" and old.tier == "measured"


def test_arcus_without_age_stays_cautious():
    f = score_arcus(0.30, 0.30, age=None)
    assert f.tier == "weak-signal"
    assert "depends heavily on age" in f.summary


# --------------------------------------------------------- iris boundary ----

def test_round_iris_reads_as_circular():
    _, iris, pupil = _masks()
    r = measure_iris_boundary(iris | pupil)
    assert r["circularity"] > 0.85
    assert r["horizontal_fill"] > 0.9


def test_encroachment_reduces_horizontal_fill():
    """A wedge growing across the cornea from the nasal side takes a bite out of
    the iris horizontally — where the lids cannot be blamed."""
    _, iris, pupil = _masks()
    m = (iris | pupil).copy()
    yy, xx = np.mgrid[0:SIZE[0], 0:SIZE[1]]
    wedge = (xx < CENTER[0] - IRIS_R * 0.45) & (np.abs(yy - CENTER[1]) < 12)
    m[wedge] = False
    r = measure_iris_boundary(m)
    base = measure_iris_boundary(iris | pupil)
    assert r["horizontal_fill"] < base["horizontal_fill"] - 0.05


def test_boundary_returns_none_for_an_empty_mask():
    assert measure_iris_boundary(np.zeros(SIZE, bool)) is None
