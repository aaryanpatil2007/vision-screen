from visionscreen.modules.behavioral import analyze_series


def steady(n=100, ear=0.28, iod=200.0, roll=0.0):
    return [ear] * n, [iod] * n, [roll] * n


def test_normal_series_no_flags():
    ears, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_squinting_flagged():
    ears = [0.28] * 60 + [0.10] * 40  # 40% of frames squinting
    _, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert "frequent squinting" in f.metrics["flags"]


def test_lean_in_flagged():
    iod = [200.0] * 80 + [260.0] * 20  # face grows 30% near the end
    ears, _, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.95)
    assert "leaning toward screen" in f.metrics["flags"]


def test_low_valid_fraction_is_inconclusive():
    ears, iod, rolls = steady()
    f = analyze_series(ears, iod, rolls, valid_fraction=0.3)
    assert f.tier == "inconclusive"
    assert f.retakes  # must tell the user how to fix it
