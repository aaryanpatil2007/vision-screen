import pytest

from visionscreen.modules.astigmatic import (
    dial_spoke_angles,
    minus_cyl_axis_from_dark_meridian,
    score_astigmatic_dial,
)


def test_dial_has_evenly_spaced_spokes():
    angles = dial_spoke_angles(n=12)
    assert len(angles) == 12
    assert angles[0] == 0
    assert angles[1] - angles[0] == pytest.approx(15.0)
    assert max(angles) < 180


def test_axis_rule_of_30():
    # Clock-dial rule: the clock hour seen darkest, x30, gives the minus-cyl axis.
    # A spoke at 90 deg (12-6 o'clock) appearing darkest => minus axis 180/0.
    assert minus_cyl_axis_from_dark_meridian(90.0) == pytest.approx(0.0)
    assert minus_cyl_axis_from_dark_meridian(0.0) == pytest.approx(90.0)
    assert minus_cyl_axis_from_dark_meridian(30.0) == pytest.approx(120.0)


def test_no_preference_means_no_astigmatism():
    f = score_astigmatic_dial(responses=[], valid_fraction=0.9, no_preference=True)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []
    assert "no meridian" in f.summary.lower() or "uniform" in f.summary.lower()


def _axis_diff(a: float, b: float) -> float:
    """Axes live on a 180-degree circle: 175 and 0 are 5 degrees apart."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def test_consistent_dark_meridian_flags_astigmatism():
    f = score_astigmatic_dial([90.0, 90.0, 75.0], valid_fraction=0.9, no_preference=False)
    assert "possible astigmatism" in f.metrics["flags"]
    assert _axis_diff(f.metrics["axis_deg"], 0.0) <= 15


def test_inconsistent_responses_downgrade_tier():
    f = score_astigmatic_dial([0.0, 90.0, 45.0], valid_fraction=0.9, no_preference=False)
    assert f.tier == "weak-signal"


def test_low_valid_fraction_inconclusive():
    f = score_astigmatic_dial([90.0], valid_fraction=0.2, no_preference=False)
    assert f.tier == "inconclusive"
    assert f.retakes
