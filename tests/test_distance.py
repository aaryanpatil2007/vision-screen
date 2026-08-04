import numpy as np
import pytest

from visionscreen.perception.distance import (
    DEFAULT_INTEROCULAR_MM,
    distance_from_interocular,
    estimate_focal_px,
    score_distance_stability,
)


def test_focal_from_calibration():
    # a 63 mm interocular spanning 200 px at 50 cm implies f = 200*500/63
    f = estimate_focal_px(interocular_px=200.0, distance_mm=500.0)
    assert f == pytest.approx(200 * 500 / DEFAULT_INTEROCULAR_MM, rel=1e-9)


def test_distance_inverts_focal():
    f = estimate_focal_px(200.0, 500.0)
    assert distance_from_interocular(200.0, f) == pytest.approx(500.0, rel=1e-9)
    # moving closer doubles the pixel span and halves the distance
    assert distance_from_interocular(400.0, f) == pytest.approx(250.0, rel=1e-9)


def test_distance_uses_person_specific_iod_when_given():
    f = estimate_focal_px(200.0, 500.0, interocular_mm=70.0)
    d = distance_from_interocular(200.0, f, interocular_mm=70.0)
    assert d == pytest.approx(500.0, rel=1e-9)


def test_rejects_degenerate_input():
    assert distance_from_interocular(0.0, 1500.0) is None
    assert estimate_focal_px(0.0, 500.0) is None
    assert estimate_focal_px(200.0, 0.0) is None


def test_stable_distance_scores_measured():
    f = score_distance_stability([500.0] * 60, nominal_mm=500.0)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []
    assert f.metrics["median_cm"] == pytest.approx(50.0, abs=0.1)


def test_drifting_distance_flagged():
    drift = list(np.linspace(500, 300, 60))   # user creeps toward the screen
    f = score_distance_stability(drift, nominal_mm=500.0)
    assert "viewing distance changed during the test" in f.metrics["flags"]


def test_wrong_distance_flagged_against_nominal():
    f = score_distance_stability([320.0] * 60, nominal_mm=500.0)
    assert "sat closer than the distance you entered" in f.metrics["flags"]
    assert f.metrics["acuity_bias_logmar"] == pytest.approx(
        np.log10(500 / 320), abs=0.02)


def test_too_few_samples_inconclusive():
    f = score_distance_stability([500.0] * 3, nominal_mm=500.0)
    assert f.tier == "inconclusive"
