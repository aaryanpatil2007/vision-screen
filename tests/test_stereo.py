import pytest

from visionscreen.modules.stereo import (
    disparity_arcsec,
    min_resolvable_arcsec,
    score_stereo,
)


def test_disparity_conversion():
    # 206265 * d / D, small-angle exact enough below a few arcmin
    assert disparity_arcsec(0.25, 600.0) == pytest.approx(85.9, abs=0.5)
    assert disparity_arcsec(0.055, 400.0) == pytest.approx(28.4, abs=0.5)
    # disparity scales as 1/D for a screen (unlike 1/D^2 for real-depth tests)
    assert disparity_arcsec(0.25, 1200.0) == pytest.approx(
        disparity_arcsec(0.25, 600.0) / 2, rel=1e-6)


def test_floor_depends_on_pixel_pitch_and_distance():
    # a 140 ppi laptop at 50 cm cannot reach the Titmus 40 arcsec rung
    floor = min_resolvable_arcsec(px_per_cm=55.1, distance_cm=50.0)
    assert floor == pytest.approx(74.8, abs=3.0)
    assert floor > 40.0
    # subpixel antialiasing buys a documented 2-4x
    assert min_resolvable_arcsec(55.1, 50.0, subpixel_factor=3.0) < 30.0


def test_normal_stereo_scored():
    trials = [{"arcsec": a, "correct": a >= 60} for a in
              (800, 400, 200, 100, 60, 40, 20)] * 2
    f = score_stereo(trials, catch_trials=[], valid_fraction=0.9)
    assert f.tier == "weak-signal"          # never 'measured' without validation
    assert f.metrics["threshold_arcsec"] == pytest.approx(60, abs=25)
    assert f.metrics["flags"] == []


def test_stereo_deficient_flagged():
    trials = [{"arcsec": a, "correct": a >= 800} for a in
              (800, 400, 200, 100, 60, 40)] * 2
    f = score_stereo(trials, catch_trials=[], valid_fraction=0.9)
    assert "reduced stereo depth perception" in f.metrics["flags"]


def test_failed_catch_trials_invalidate_result():
    """Catch trials carry zero disparity — passing them means the subject is
    reading a non-stereo cue, so the whole result must be voided."""
    trials = [{"arcsec": a, "correct": True} for a in (800, 400, 200, 100, 40)] * 2
    catch = [{"correct": True}, {"correct": True}, {"correct": True}]
    f = score_stereo(trials, catch_trials=catch, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert "non-stereo cue" in f.summary.lower()
    assert "threshold_arcsec" not in f.metrics


def test_chance_level_catch_trials_are_fine():
    trials = [{"arcsec": a, "correct": a >= 100} for a in
              (800, 400, 200, 100, 60, 40)] * 2
    catch = [{"correct": True}, {"correct": False}, {"correct": False},
             {"correct": False}]
    f = score_stereo(trials, catch_trials=catch, valid_fraction=0.9)
    assert f.tier != "inconclusive"


def test_floor_limited_result_is_reported_honestly():
    trials = [{"arcsec": a, "correct": True} for a in (800, 400, 200, 100)] * 2
    f = score_stereo(trials, catch_trials=[], valid_fraction=0.9,
                     display_floor_arcsec=100.0)
    assert "at or better than" in f.summary.lower()
    assert f.metrics["display_floor_arcsec"] == 100.0


def test_too_few_trials_inconclusive():
    f = score_stereo([{"arcsec": 400, "correct": True}], [], 0.9)
    assert f.tier == "inconclusive"
    assert f.retakes
