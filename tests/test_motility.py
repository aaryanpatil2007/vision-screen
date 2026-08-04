import numpy as np
import pytest

from visionscreen.modules.motility import (
    detect_saccades,
    pursuit_gain,
    score_motility,
)


def _pursuit_signal(n=180, fps=30.0, gain=1.0, lag_s=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    target = 0.5 + 0.35 * np.sin(2 * np.pi * 0.25 * t)
    eye = 0.5 + gain * 0.35 * np.sin(2 * np.pi * 0.25 * (t - lag_s))
    eye = eye + rng.normal(0, noise, n)
    return list(eye), list(target), list(t)


def test_pursuit_gain_unity_for_perfect_tracking():
    eye, target, ts = _pursuit_signal(gain=1.0)
    g = pursuit_gain(eye, target, ts)
    assert g["gain"] == pytest.approx(1.0, abs=0.1)
    assert g["lag_s"] == pytest.approx(0.0, abs=0.06)


def test_pursuit_gain_low_when_undershooting():
    eye, target, ts = _pursuit_signal(gain=0.45)
    g = pursuit_gain(eye, target, ts)
    assert g["gain"] == pytest.approx(0.45, abs=0.12)


def test_detects_saccades_in_step_signal():
    fps = 60.0
    t = np.arange(120) / fps
    eye = np.where(t < 1.0, 0.3, 0.7)  # single step
    s = detect_saccades(list(eye), list(t))
    assert len(s) == 1
    assert s[0]["amplitude"] == pytest.approx(0.4, abs=0.05)
    assert s[0]["peak_velocity"] > 1.0


def test_no_saccades_in_smooth_signal():
    eye, _, ts = _pursuit_signal(gain=1.0)
    assert detect_saccades(eye, ts) == []


def test_score_normal_motility():
    eye, target, ts = _pursuit_signal(gain=0.98)
    f = score_motility(eye, eye, target, ts, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_low_gain_flagged():
    slow, target, ts = _pursuit_signal(gain=0.35)
    f = score_motility(slow, slow, target, ts, valid_fraction=0.9)
    assert "reduced smooth-pursuit gain" in f.metrics["flags"]


def test_interocular_asymmetry_flagged():
    good, target, ts = _pursuit_signal(gain=1.0, seed=1)
    bad, _, _ = _pursuit_signal(gain=0.3, seed=2)
    f = score_motility(good, bad, target, ts, valid_fraction=0.9)
    assert "asymmetric eye movement" in f.metrics["flags"]


def test_short_recording_inconclusive():
    f = score_motility([0.5] * 5, [0.5] * 5, [0.5] * 5, [0.0] * 5, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert f.retakes
