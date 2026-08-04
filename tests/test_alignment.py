import math

import numpy as np
import pytest

from visionscreen.modules.alignment import (
    AlignmentFrame,
    hirschberg_pd,
    pursuit_conjugacy,
    reflex_decentration_mm,
    score_alignment,
)


def test_decentration_scale():
    # iris 80 px wide = 11.7 mm → 1 mm ≈ 6.84 px
    dx, dy = reflex_decentration_mm((106.84, 100.0), (100.0, 100.0), 80.0)
    assert dx == pytest.approx(1.0, rel=0.01)
    assert dy == pytest.approx(0.0, abs=1e-9)


def test_hirschberg_ratio():
    assert hirschberg_pd((1.0, 0.0)) == pytest.approx(18.0)
    assert hirschberg_pd((0.0, 0.0)) == 0.0


def frames(dec_left, dec_right, n=50):
    return [AlignmentFrame(dec_left, dec_right) for _ in range(n)]


def test_symmetric_offset_not_flagged():
    # both eyes offset identically = camera angle, not strabismus
    f = score_alignment(frames((0.8, 0.0), (0.8, 0.0)), None, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_asymmetry_flagged_with_pd():
    f = score_alignment(frames((0.0, 0.0), (1.5, 0.0)), None, valid_fraction=0.9)
    assert "possible eye misalignment" in f.metrics["flags"]
    assert f.metrics["deviation_pd"] == pytest.approx(27.0, abs=1.0)


def test_low_valid_fraction_inconclusive():
    f = score_alignment(frames((0.0, 0.0), (0.0, 0.0)), None, valid_fraction=0.2)
    assert f.tier == "inconclusive"
    assert f.retakes


def sinusoid(n=60, phase=0.0, amp=1.0):
    return [amp * math.sin(2 * math.pi * i / 30 + phase) for i in range(n)]


def test_conjugate_pursuit_high_correlation():
    dot = sinusoid()
    res = pursuit_conjugacy(sinusoid(), sinusoid(), dot)
    assert res is not None
    assert res.conjugacy > 0.99
    f = score_alignment(frames((0.0, 0.0), (0.0, 0.0)), res, valid_fraction=0.9)
    assert "poor pursuit conjugacy" not in f.metrics["flags"]


def test_nonconjugate_pursuit_flagged():
    dot = sinusoid()
    rng = np.random.default_rng(1)
    flat = list(rng.normal(0, 0.05, 60))  # one eye doesn't follow
    res = pursuit_conjugacy(sinusoid(), flat, dot)
    assert res.conjugacy < 0.8
    f = score_alignment(frames((0.0, 0.0), (0.0, 0.0)), res, valid_fraction=0.9)
    assert "poor pursuit conjugacy" in f.metrics["flags"]


def test_too_few_samples_returns_none():
    assert pursuit_conjugacy([0.1] * 5, [0.1] * 5, [0.1] * 5) is None


def test_static_dot_returns_none():
    # a dot that never moved cannot measure pursuit — must not flag anything
    assert pursuit_conjugacy(sinusoid(), sinusoid(), [0.5] * 60) is None


def test_pursuit_only_when_reflex_missing():
    # real webcams often lose the corneal reflex; pursuit must survive alone
    dot = sinusoid()
    res = pursuit_conjugacy(sinusoid(), sinusoid(), dot)
    f = score_alignment([], res, valid_fraction=0.9)
    assert f.tier == "weak-signal"
    assert f.metrics["conjugacy"] > 0.99
    assert "poor pursuit conjugacy" not in f.metrics["flags"]
    assert "Hirschberg" in f.summary  # explains what was skipped and why
