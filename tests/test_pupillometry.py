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
            d = baseline - amplitude * np.exp(-since / 0.9) * (1 - np.exp(-since / 0.08))
        diam.append(d + rng.normal(0, noise))
        ts.append(t)
    return PupilTrace(ts=ts, diameter_mm=diam, flash_ts=flash_at)


def test_constriction_metrics_recovered():
    m = constriction_metrics(synth_trace())
    assert m is not None
    assert m["baseline_mm"] == pytest.approx(4.0, abs=0.15)
    assert m["constriction_pct"] > 15
    assert 0.05 < m["latency_s"] < 0.9


def test_no_response_detected():
    flat = PupilTrace(ts=[i / 30 for i in range(90)], diameter_mm=[4.0] * 90, flash_ts=0.5)
    assert constriction_metrics(flat)["constriction_pct"] < 5


def test_short_trace_rejected():
    tr = PupilTrace(ts=[0.0, 0.1], diameter_mm=[4.0, 4.0], flash_ts=0.05)
    assert constriction_metrics(tr) is None


def test_score_normal_bilateral():
    f = score_pupillometry(synth_trace(seed=1), synth_trace(seed=2), 0.9, fps=60.0)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_never_claims_rapd_from_bilateral_flash():
    """A screen flash reaches both retinas, so response asymmetry cannot mean
    an afferent defect. The module must not say otherwise."""
    normal = synth_trace(amplitude=1.2, seed=3)
    weak = synth_trace(amplitude=0.25, seed=4)
    f = score_pupillometry(normal, weak, 0.9, fps=60.0)
    assert "asymmetric pupil response" not in f.metrics["flags"]
    assert "afferent" in f.summary.lower()      # states the limitation explicitly
    assert "rapd" not in " ".join(f.metrics["flags"]).lower()


def test_anisocoria_flagged_only_above_one_mm():
    """<1 mm is common in normals (41% show >=0.4 mm at some sitting)."""
    small = score_pupillometry(synth_trace(baseline=4.0), synth_trace(baseline=4.5),
                               0.9, fps=60.0)
    assert "unequal pupil sizes (anisocoria)" not in small.metrics["flags"]
    big = score_pupillometry(synth_trace(baseline=4.0), synth_trace(baseline=5.3),
                             0.9, fps=60.0)
    assert "unequal pupil sizes (anisocoria)" in big.metrics["flags"]
    assert big.metrics["anisocoria_mm"] == pytest.approx(1.3, abs=0.15)


def test_latency_withheld_at_low_fps():
    """30 fps quantization (~9.6 ms SD) swamps inter-eye latency asymmetry."""
    at30 = score_pupillometry(synth_trace(), synth_trace(), 0.9, fps=30.0)
    assert "latency_s" not in at30.metrics["left"]
    assert "latency_withheld" in at30.metrics
    at60 = score_pupillometry(synth_trace(fps=60.0), synth_trace(fps=60.0), 0.9, fps=60.0)
    assert "latency_s" in at60.metrics["left"]


def test_no_response_either_eye_flagged():
    flat = PupilTrace(ts=[i / 30 for i in range(90)], diameter_mm=[4.0] * 90, flash_ts=0.5)
    f = score_pupillometry(flat, flat, 0.9)
    assert "no measurable light response" in f.metrics["flags"]


def test_low_valid_fraction_inconclusive():
    tr = synth_trace()
    f = score_pupillometry(tr, tr, 0.1)
    assert f.tier == "inconclusive"
    assert f.retakes
