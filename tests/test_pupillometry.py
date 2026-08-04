import numpy as np
import pytest

from visionscreen.modules.pupillometry import (
    PupilTrace,
    constriction_metrics,
    score_pupillometry,
)


def synth_trace(baseline=4.0, amplitude=1.2, latency_s=0.25, fps=30.0, dur=3.0,
                flash_at=0.5, noise=0.0, seed=0):
    """Pupil diameter (mm) over time: steady, then constrict, then redilate."""
    rng = np.random.default_rng(seed)
    ts, diam = [], []
    n = int(dur * fps)
    for i in range(n):
        t = i / fps
        d = baseline
        since = t - (flash_at + latency_s)
        if since >= 0:
            # fast constriction then slow recovery
            d = baseline - amplitude * np.exp(-since / 0.9) * (1 - np.exp(-since / 0.08))
        diam.append(d + rng.normal(0, noise))
        ts.append(t)
    return PupilTrace(ts=ts, diameter_mm=diam, flash_ts=flash_at)


def test_constriction_metrics_recovered():
    tr = synth_trace()
    m = constriction_metrics(tr)
    assert m is not None
    assert m["baseline_mm"] == pytest.approx(4.0, abs=0.15)
    assert m["constriction_pct"] > 15
    assert 0.05 < m["latency_s"] < 0.9


def test_no_response_detected():
    flat = PupilTrace(ts=[i / 30 for i in range(90)], diameter_mm=[4.0] * 90, flash_ts=0.5)
    m = constriction_metrics(flat)
    assert m["constriction_pct"] < 5


def test_score_normal_bilateral():
    left, right = synth_trace(seed=1), synth_trace(seed=2)
    f = score_pupillometry(left, right, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_rapd_flagged_on_asymmetric_response():
    normal = synth_trace(amplitude=1.2, seed=3)
    weak = synth_trace(amplitude=0.25, seed=4)   # afferent defect in one eye
    f = score_pupillometry(normal, weak, valid_fraction=0.9)
    assert "asymmetric pupil response" in f.metrics["flags"]


def test_no_response_either_eye_flagged():
    flat = PupilTrace(ts=[i / 30 for i in range(90)], diameter_mm=[4.0] * 90, flash_ts=0.5)
    f = score_pupillometry(flat, flat, valid_fraction=0.9)
    assert "no measurable light response" in f.metrics["flags"]


def test_low_valid_fraction_inconclusive():
    tr = synth_trace()
    f = score_pupillometry(tr, tr, valid_fraction=0.1)
    assert f.tier == "inconclusive"
    assert f.retakes


def test_short_trace_rejected():
    tr = PupilTrace(ts=[0.0, 0.1], diameter_mm=[4.0, 4.0], flash_ts=0.05)
    assert constriction_metrics(tr) is None
