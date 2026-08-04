import pytest

from visionscreen.modules.contrast import (
    PELLI_ROBSON_STEP,
    contrast_to_luminance_pair,
    log_cs_to_weber,
    score_contrast,
    triplet_levels,
)


def test_pelli_robson_step_is_standard():
    # Pelli-Robson: 0.15 log unit steps, triplets of 3 letters
    assert PELLI_ROBSON_STEP == pytest.approx(0.15)


def test_triplet_levels_descend_by_step():
    levels = triplet_levels(start_log_cs=0.0, n=8)
    assert levels[0] == pytest.approx(0.0)
    assert levels[1] - levels[0] == pytest.approx(PELLI_ROBSON_STEP)
    assert levels[-1] == pytest.approx(0.15 * 7)


def test_log_cs_to_weber_contrast():
    # log CS 2.0 => contrast 1% => Weber 0.01
    assert log_cs_to_weber(2.0) == pytest.approx(0.01)
    assert log_cs_to_weber(0.0) == pytest.approx(1.0)


def test_luminance_pair_produces_valid_8bit_levels():
    fg, bg = contrast_to_luminance_pair(1.5, background=255)
    assert 0 <= fg <= 255 and bg == 255
    assert fg < bg
    # higher log CS => fainter letter => closer to background
    fg2, _ = contrast_to_luminance_pair(2.0, background=255)
    assert fg2 > fg


def test_score_normal_vision():
    # gets everything right down to 1.95 (normal adult ≈ 1.75-2.0)
    trials = [{"log_cs": round(0.15 * i, 2), "correct": 0.15 * i <= 1.95}
              for i in range(14)]
    f = score_contrast(trials, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["log_cs"] == pytest.approx(1.95, abs=0.16)
    assert f.metrics["flags"] == []


def test_score_reduced_contrast_flagged():
    trials = [{"log_cs": round(0.15 * i, 2), "correct": 0.15 * i <= 1.05}
              for i in range(14)]
    f = score_contrast(trials, valid_fraction=0.9)
    assert "reduced contrast sensitivity" in f.metrics["flags"]


def test_too_few_trials_inconclusive():
    f = score_contrast([{"log_cs": 0.0, "correct": True}], valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert f.retakes
