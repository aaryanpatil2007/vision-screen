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


def test_ten_pd_deviation_is_flagged():
    """Clinically significant strabismus starts near 10 PD — it must not be missed."""
    dec = (10.0 / 18.0, 0.0)   # 10 prism diopters of asymmetry
    f = score_alignment(frames((0.0, 0.0), dec), None, valid_fraction=0.9)
    assert "possible eye misalignment" in f.metrics["flags"]
    assert f.metrics["deviation_pd"] == pytest.approx(10.0, abs=0.5)


def test_small_physiological_offset_not_flagged():
    dec = (2.0 / 18.0, 0.0)    # ~2 PD, within normal phoria/kappa variation
    f = score_alignment(frames((0.0, 0.0), dec), None, valid_fraction=0.9)
    assert "possible eye misalignment" not in f.metrics["flags"]


def test_implausible_deviation_rejected_not_reported():
    """Real-photograph validation produced readings up to 163 PD on normal
    faces — stray window/lamp reflections, not measurements."""
    huge = (163.0 / 18.0, 0.0)
    f = score_alignment(frames((0.0, 0.0), huge, n=30), None, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert "deviation_pd" not in f.metrics
    assert f.metrics["rejected_reason"] == "implausible magnitude"
    assert f.retakes


def test_unstable_reflex_across_frames_rejected():
    """A real deviation is stable; uncorrelated stray highlights jitter."""
    import numpy as _np
    rng = _np.random.default_rng(0)
    noisy = [
        AlignmentFrame((0.0, 0.0), (float(rng.normal(0, 1.2)), float(rng.normal(0, 1.2))))
        for _ in range(30)
    ]
    f = score_alignment(noisy, None, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert f.metrics["rejected_reason"] == "unstable across frames"


def test_stable_real_deviation_still_reported():
    """The guards must not suppress a genuine, consistent deviation."""
    f = score_alignment(frames((0.0, 0.0), (20.0 / 18.0, 0.0), n=30), None,
                        valid_fraction=0.9)
    assert f.tier == "measured"
    assert "possible eye misalignment" in f.metrics["flags"]
    assert f.metrics["deviation_pd"] == pytest.approx(20.0, abs=1.0)


def test_single_frame_magnitude_is_provisional():
    f = score_alignment(frames((0.0, 0.0), (20.0 / 18.0, 0.0), n=1), None,
                        valid_fraction=0.9)
    assert f.tier == "weak-signal"
    assert "provisional" in f.summary


def test_expected_error_matches_measured_aggregation():
    """The frame requirement is set from bench_real_hirschberg, which measured
    1.09 PD at 40 frames on real eye images; the model must reproduce it."""
    from visionscreen.modules.alignment import (
        MIN_FRAMES_FOR_MAGNITUDE, expected_magnitude_error_pd,
    )
    assert expected_magnitude_error_pd(40) == pytest.approx(1.09, abs=0.15)
    assert expected_magnitude_error_pd(1) == pytest.approx(5.35, abs=0.6)
    # the chosen minimum must keep magnitude error well inside the ~5 PD
    # interexaminer agreement of the prism cover test
    assert expected_magnitude_error_pd(MIN_FRAMES_FOR_MAGNITUDE) < 2.0
    # and error must fall with more frames
    assert expected_magnitude_error_pd(60) < expected_magnitude_error_pd(20)


def test_reported_deviation_carries_its_own_error_bar():
    f = score_alignment(frames((0.0, 0.0), (20.0 / 18.0, 0.0), n=40), None,
                        valid_fraction=0.9)
    assert f.metrics["expected_error_pd"] < 2.0
    assert f.tier == "measured"


def test_few_frames_downgraded_even_when_clean():
    """10 frames implies ~2 PD error — not enough to quote a magnitude."""
    f = score_alignment(frames((0.0, 0.0), (20.0 / 18.0, 0.0), n=10), None,
                        valid_fraction=0.95)
    assert f.tier == "weak-signal"
    assert "provisional" in f.summary
