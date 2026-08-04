import numpy as np
import pytest

from visionscreen.modules.photoref import (
    fit_srx,
    measure_reflex,
    score_photoref,
)
from visionscreen.synth.photoref import render_reflex

PX_PER_M = 8000.0
KW = dict(e_m=0.005, d_m=0.5, px_per_m=PX_PER_M)


def roundtrip(S, C=0.0, axis=0.0, noise=0.0):
    img, truth = render_reflex(32, S=S, C=C, axis_deg=axis, noise_sigma=noise, px_per_m=PX_PER_M)
    return measure_reflex(img, truth["center_px"], truth["pupil_radius_px"], **KW)


@pytest.mark.parametrize("S", [-4.0, -3.0, -2.0, 2.0, 3.0, 4.0])
def test_sphere_round_trip(S):
    est = roundtrip(S)
    assert est is not None
    S_est, C_est, _ = est
    assert S_est == pytest.approx(S, abs=0.5)
    assert C_est == pytest.approx(0.0, abs=0.5)


def test_cylinder_axis_recovered():
    # S=-4 keeps every meridian's |A| ∈ [2,4], outside the 1.25 D dead zone;
    # with smaller |S| the flattest meridians vanish and C reads low (physical).
    est = roundtrip(-4.0, C=2.0, axis=45.0)
    assert est is not None
    S_est, C_est, axis_est = est
    assert C_est == pytest.approx(2.0, abs=0.75)
    assert min(abs(axis_est - 45.0), 180 - abs(axis_est - 45.0)) <= 15.0


def test_dead_zone_returns_none():
    assert roundtrip(0.5) is None


def test_fit_srx_constant_profile():
    thetas = np.arange(-60, 61, 5, dtype=float)
    profile = [(t, 3.0) for t in thetas]
    S_abs, C, _ = fit_srx(profile)
    assert S_abs == pytest.approx(3.0, abs=0.1)
    assert C == pytest.approx(0.0, abs=0.1)


def test_score_measured_when_consistent():
    ests = [(-2.0 + 0.05 * i, 0.2, 90.0) for i in range(10)]
    f = score_photoref(ests, dead_frames=0, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["sphere_d"] == pytest.approx(-1.8, abs=0.3)
    assert "screening estimate" in f.summary


def test_score_dead_zone_dominated():
    f = score_photoref([], dead_frames=20, valid_fraction=0.9)
    assert f.tier == "weak-signal"
    assert "dead zone" in f.summary or "within" in f.summary


def test_score_no_data_inconclusive():
    f = score_photoref([], dead_frames=0, valid_fraction=0.2)
    assert f.tier == "inconclusive"
    assert f.retakes


def test_unstable_estimates_rejected_not_averaged():
    """18% of real crops with no crescent still yield an estimate; scattered
    values must be rejected rather than averaged into a confident number."""
    noisy = [(-6.0, 0.2, 10.0), (1.5, 0.1, 90.0), (-2.0, 0.3, 45.0),
             (4.0, 0.2, 120.0), (-5.5, 0.1, 30.0), (0.5, 0.2, 70.0)]
    f = score_photoref(noisy, dead_frames=0, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert f.metrics["rejected_reason"] == "unstable across frames"
    assert "sphere_d" not in f.metrics
    assert f.retakes


def test_consistent_estimates_still_reported():
    steady = [(-2.0 + 0.05 * i, 0.2, 90.0) for i in range(8)]
    f = score_photoref(steady, dead_frames=0, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["sphere_d"] == pytest.approx(-1.83, abs=0.2)


def test_radius_error_is_proportional_not_additive():
    """Pupil radius enters w = 2r - e/(d|A|), so a radius error scales the
    diopter estimate rather than offsetting it. This is why the measured
    real-image error grew with defocus (0.46 D at 2 D, 1.40 D at 4 D with a
    population constant) and why using the measured radius cut it 3.8x."""
    from visionscreen.modules.photoref import invert_width

    px_per_m, e_m, d_m = 8000.0, 0.005, 0.5
    true_r, wrong_r = 32.0, 32.0 * 1.4      # 40% too large
    ratios = []
    for w_frac in (0.3, 0.5, 0.7):
        w = 2 * true_r * w_frac
        a_true = invert_width(w, true_r, e_m, d_m, px_per_m)
        a_wrong = invert_width(w, wrong_r, e_m, d_m, px_per_m)
        assert a_true and a_wrong
        ratios.append(a_wrong / a_true)
    # a proportional error shows up as a roughly constant RATIO across
    # magnitudes, which an additive offset would not produce
    assert max(ratios) / min(ratios) < 2.0, ratios
    assert all(r < 1.0 for r in ratios), ratios
