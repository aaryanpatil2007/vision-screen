import math

import pytest

from visionscreen.modules.acuity import Staircase, letter_height_px, score_trials


def test_letter_height_physics():
    # logMAR 0.0 at 50 cm: 5 arcmin → height = 2*50*tan(2.5') ≈ 0.0727 cm
    px = letter_height_px(0.0, distance_cm=50.0, px_per_cm=37.8)
    assert px == pytest.approx(2 * 50 * math.tan(math.radians(5 / 60 / 2)) * 37.8, rel=1e-3)
    # one logMAR unit = 10x the size
    assert letter_height_px(1.0, 50.0, 37.8) == pytest.approx(px * 10, rel=1e-3)


def simulate(true_logmar: float) -> Staircase:
    s = Staircase()
    while not s.done:
        s.record(s.current() >= true_logmar)  # ideal observer: correct iff letter big enough
    return s


def test_staircase_converges_to_true_threshold():
    s = simulate(0.4)
    assert s.threshold() == pytest.approx(0.4, abs=0.15)


def test_staircase_terminates_within_30_trials():
    s = Staircase()
    for _ in range(30):
        if s.done:
            break
        s.record(False)
    assert s.done


def make_trials(n: int, logmar: float = 0.3) -> list[dict]:
    return [{"logmar": logmar, "shown": "up", "answered": "up"} for _ in range(n)]


def test_floor_reported_as_at_or_better():
    trials = make_trials(20, logmar=-0.3)  # perfect run down at the floor
    f = score_trials(trials)
    assert "at or better than" in f.summary
    assert f.metrics["logmar"] == -0.3


def test_score_trials_tiers():
    assert score_trials(make_trials(20)).tier == "measured"
    assert score_trials(make_trials(10)).tier == "weak-signal"
    f = score_trials(make_trials(3))
    assert f.tier == "inconclusive"
    assert f.metrics == {} or "logmar" not in f.metrics
